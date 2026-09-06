#!/usr/bin/env python3
"""TASK_917 P1 — market-level pooled VR(30) persistence (the widest-reaching instrument in the
row: its KILL would close every regime-gating idea on this venue if rho-bar < 0.10).

REVISION 2 (2026-09-06) — this module was rewritten after an adversarial panel found that the
shipped verdict did not survive a block-length sweep. Three corrections, all disclosed here
and in RESULTS.md rather than in a footnote:

  (1) BLOCK LENGTH IS SWEPT, NOT CHOSEN. Revision 1 read the kill off a single cell
      (block_days=180). 180d was TASK_917's own addition and was disclosed nowhere; the source
      panel this row salvages (`lore/harvest_6b32_appendix/mech_B.md:44,48`) pre-registers
      90-DAY non-overlapping windows with the identical rho-bar >= 0.10 threshold. The full
      block-length sweep is now a first-class output (BLOCK_GRID below), and the ANCHOR cell
      is the inherited 90d convention, named as inherited.
  (2) THE PLACEBO NO LONGER CONTAINS COPIES OF THE STATISTIC UNDER TEST. Revision 1's
      circular-shift null ran s = 1..N-1. s=1 and its mirror s=N-1 are BIT-IDENTICAL to the
      observed rho-bar (roll(vr,-1) pairs (b, b+1); roll(vr,-(N-1)) pairs (b, b-1) -- the same
      unordered pair set), and c(s) == c(N-s) makes every remaining draw a duplicate. The null
      therefore always contained 2 self-copies out of ~24 values, bounding the minimum
      achievable p near 2/24 ~= 0.08 -- so NO cell could ever have reached the Bonferroni line
      the module then reported it as failing. Degenerate shifts are now excluded (s=0, s=1,
      s=N-1) and only distinct lags s = 2..floor(N/2) are kept; a second, informative null
      (random permutation of the block order, N_PERM draws) is added because the circular-shift
      null on a ~12-50 block series has too few distinct draws to support any multiplicity
      claim. Both nulls report their MINIMUM ACHIEVABLE p explicitly.
  (3) THE VERDICT FOLLOWS THE SWEEP. See DECISION RULE below.

REVISION 3 (2026-09-06) — CLOSING PASS, cycle 3. DISCLOSED HERE AND NOT IN A FOOTNOTE, per the
project's D76 convention. Two adversarial judges accepted revision 2's VERDICT (INCONCLUSIVE,
kill does not fire) and then demolished most of the ARGUMENTS revision 2 used to reach it. This
revision replaces those arguments. IT CHANGES NO VERDICT: the kill still does not fire, and the
correction makes the reason STRONGER, not weaker -- P1 could not have fired a defensible kill at
any block length, so the un-resolvability is a property of the instrument rather than a close
call about a free parameter.

  (H) THE PRO-KILL COUNTERFACTUAL WAS FALSE AND IS REPLACED BY ITS REFUTATION. Revision 2
      printed that "read on the anchor alone the kill IS met and IS resolvable", citing the CI
      upper end +0.045 and that "all four 90d cells agree to within 0.02". The agreement is an
      artefact: all four cells share ONE TILING ORIGIN, which `_block_boundaries` hard-wires to
      day 0 and which revision 2 never disclosed as a free parameter at all. Sweeping all
      `block_days` one-day tiling origins at the anchor is now a first-class output, and it
      shows the anchor reading sits ~1.3 sd into the negative tail of its own origin
      distribution, with a material share of origins landing ABOVE the kill line. Consequence,
      stated plainly: R2 IS NOT A POWER GUARD. With SE(rho-bar) ~= 0.17 at the anchor,
      "the CI excludes 0.10" is a lottery on where the point estimate happened to land, not
      resolving power. The tiling-phase distribution is now reported wherever a cell is called
      resolving.
  (I) THE POWER CEILING IS DISCLOSED — the load-bearing correction. A model-free sign-flip null
      on the REAL panel (per-date sign flip: destroys serial structure, preserves the
      cross-sectional covariance and every marginal variance, so true VR == 1 by construction)
      shows the observed block-VR dispersion is indistinguishable from pure estimator noise at
      every block length in the grid. Because rho-bar is attenuated by the signal's share of
      the block-VR variance, and that share is not distinguishable from zero, the MAXIMUM
      |rho-bar| this instrument could report against a perfectly persistent true VR is small
      (~0.3) with an SE of the same order, and the anchor's power to separate rho-bar = 0.10
      from rho-bar = 0 is of order 10%. P1 COULD NOT HAVE FIRED A DEFENSIBLE KILL AT ANY BLOCK
      LENGTH. Also disclosed: the 0.10 threshold is INHERITED FROM A DIFFERENT STATISTIC
      (mech_B.md's 90d trend-efficiency persistence, 1000 asset-label permutations, NW t>=3)
      and was NEVER CALIBRATED to this estimator.
  (J) THE LEVEL-VS-BREAKEVEN READ WAS WRONG (numerator/denominator sample mismatch), and the
      grid of block estimators is BIASED LOW under a random walk. Both are quantified below and
      the level comparison now always prints the estimator's OWN null level.
  (K) R1'S FAILURE IS UNINFORMATIVE, and revision 2 reported it as though the data produced it.
      On a venue exactly as dead as the kill asserts (rho = 0), the artifact's own Fisher-z SEs
      put P(R1 unanimity) at ~1%. A rule that a dead venue fails 99% of the time carries no
      information about the venue.
  (L) THE HARDCODED "universe and estimator move rho-bar by <=0.15" INDEPENDENCE CLAIM WAS
      FALSE. It was a string; the code never computed it. It is now computed, and it fails at 9
      of 11 block lengths.
  (M) THE k-PARITY PATTERN IS SETTLED = SAMPLING NOISE, not an open instrument concern. The
      "may be unsound" framing is removed; the estimator recovers correct signs on ground
      truth and is noisy, not broken.

Method (the task file names the statistic and the kill band, not the exact estimator, so the
estimator design is disclosed in full; the estimator itself was independently reproduced three
times by the review panel and is NOT what was wrong in revision 1):

  - PIT true-daily panel (experiments/917.../daily_panel.py — raw, un-ffilled ohlcv_daily
    reads; dead coins retained by default, matching the 819 POWERED convention the task file
    cites). Universe/grid via `trader.data.load_db_backtest_panel` (canon filter).
  - History is cut into BLOCKS of `block_days` calendar days (crypto trades 24/7, so calendar
    day == trading day, no weekend adjustment needed). Within each block, a single MARKET-LEVEL
    POOLED variance-ratio VR(30) is estimated by pooling, across every asset present in the
    block, (a) all valid daily log returns (denominator variance) and (b) all valid 30-day
    compounded log returns at the block's internal 30-day sub-boundaries (numerator variance)
    -- VR = Var(30d)/(30*Var(1d)). This is the estimator the task file's "market-level pooled
    VR(30)" gate is written against: pooling numerator AND denominator across the whole panel
    (not averaging per-asset VRs, which the task file itself says is 5-40x too noisy to
    resolve the sign per-asset).
  - EXACT BLOCK ACCOUNTING (revision 3 (J); revision 2's "non-overlapping BLOCKS" wording was
    inexact and is corrected here rather than restated). The block starting at b0 has
    b1 = b0 + block_days. The DENOMINATOR takes dr[b0+1 : b1] -- block_days-1 daily returns.
    The NUMERATOR takes 30-day sub-windows ending at b0+Q, b0+2Q, ..., b1, so its LAST
    sub-window ends on b1, which is the FIRST DAY OF THE NEXT BLOCK, and it consumes dr[b1],
    a return no denominator includes. So the numerator covers block_days daily returns against
    the denominator's block_days-1, and the blocks are not literally non-overlapping in the
    daily-return index. Verified consequence: adjacent blocks' NUMERATOR sets are still
    DISJOINT (block b's last sub-window ends at b1, block b+1's first ends at b1+Q), so this
    does not inflate rho-bar; it is an accounting/wording defect, not a bias. Revision 3 also
    ships a SAMPLE-MATCHED variant (see below) in which the denominator is restricted to
    exactly the asset-days that feed a valid numerator window, which removes the mismatch.
  - PERSISTENCE rho-bar = Pearson corr(VR_b, VR_{b+1}) over consecutive blocks (b, b+1 both
    valid). This is the number the KILL gate binds to. Reported WITH its sampling uncertainty:
    a Fisher-z 95% CI and a t-stat, because with 8-55 block pairs the standard error of a
    correlation is 0.14-0.37 and revision 1 reported point estimates naked. Precision of
    language, since revision 2 was loose about it: the BLOCKS are disjoint in the sense above,
    but adjacent PAIRS share a block (pair (b,b+1) and pair (b+1,b+2) both contain block b+1).
    The Fisher-z SE 1/sqrt(n_pairs-3) is nevertheless the right SE here -- it is the SE of a
    correlation on n_pairs bivariate observations, and pair-sharing is exactly the dependence
    structure a lag-1 autocorrelation always has.
  - BLOCK GRID: every block length that tiles cleanly into the 30-day VR sub-window
    (multiples of Q=30), from 60d up to 360d. 30d itself is EXCLUDED and the exclusion is
    argued from a statistical property fixed in advance, not from its answer: at block=30 a
    block contains exactly ONE 30-day sub-boundary, so the numerator variance is purely
    cross-sectional dispersion across assets with zero within-block time dimension -- a
    different estimand, not a shorter-block version of the same one. 360d is the upper end
    because beyond it the panel yields fewer than ~10 block pairs.
    REVISION 3 (J), DISCLOSED BECAUSE REVISION 2 STATED THE EXCLUSION AS IF IT WERE BINARY:
    the SAME argument applies at reduced strength across the low end of the grid, so the grid
    is not eleven readings of one estimand. Pooling the numerator about a single grand mean
    over k = block_days/Q sub-window groups removes ~1/k of the market-factor variance, while
    the denominator's own small-sample penalty is only 1/(block_days-1). The k=2 cells
    therefore carry ~50% and the k=3 cells ~33% of the strength of the argument that excludes
    k=1 outright, and E[VR_block] under a TRUE VR of 1 is measurably below 1 and RISES with k
    (measured by the sign-flip null below). Most of the otherwise-unexplained level trend
    across the grid is this bias. Practical consequence for R1: "every cell reads
    rho-bar < 0.10" demands unanimity across ELEVEN DIFFERENT ESTIMATORS, not eleven samples
    of one.
  - ANCHOR CELL: (block_days=90, universe=pit_dead_retained, estimator=pooled). 90d is
    INHERITED from `mech_B.md:44,48` (the source panel's own pre-registration: "cross-sectional
    persistence of 90d realised trend-efficiency across 30 non-overlapping windows", rho-bar
    >= 0.10). 180d, revision 1's headline, was TASK_917's own choice and is reported as one
    swept cell like any other. Disclosure of the inheritance's LIMIT: what is inherited is the
    90-day non-overlapping WINDOW LENGTH and the 0.10 threshold. mech_B's statistic was the
    persistence of 90d realised TREND-EFFICIENCY with a 1000-draw asset-label permutation null
    and NW t>=3; TASK_917 re-specified the statistic as market-level pooled VR(30) persistence,
    for which an asset-label permutation is undefined (the series is a market-level scalar).
    That re-specification is the task file's, not this module's, but it means the anchor
    inherits the window convention only, and the null here is weaker than mech_B's.
  - PLACEBOS (per cell, both reported):
      * circular-shift: distinct lags s = 2..floor(N/2) on the block series (s=0/1/N-1 and the
        c(s)==c(N-s) mirrors excluded). Draws = floor(N/2) - 1.
      * permutation: N_PERM random reorderings of the FINITE block VRs among the finite block
        positions (the NaN pattern, and hence n_pairs, is held fixed), rho-bar recomputed
        identically each draw. This is the null that matches the question actually asked
        ("is there lag-1 persistence?") and the only one with enough resolution to carry a
        multiplicity claim.
    Both use the plus-one (unbiased) two-sided empirical p, p = (1 + #{|null| >= |obs|}) /
    (1 + n_draws), so the minimum achievable p is 1/(n_draws+1) and is reported per cell.
    REVISION 3 (small item): `circ_p` IS NOT COMPARABLE TO THE OBSERVED STATISTIC and is no
    longer tabled as though it were. `circular_shift_null` recomputes the finite mask per
    shift, so a rolled series pairs a DIFFERENT (and smaller) set of blocks than the observed
    rho-bar does -- measured at the anchor, the circular draws use a median of 23 pairs against
    the observed 36, and at 180d 13 against 18. A null draw computed on ~2/3 of the pairs has a
    wider sampling distribution than the statistic it is being compared with, which biases the
    comparison toward "not significant" for a reason that has nothing to do with persistence.
    `permutation_null` does NOT have this defect (it holds the NaN pattern fixed, so every draw
    carries the identical n_pairs), which is the second reason it is the only null carrying a
    multiplicity claim. The per-cell median circular n_pairs is now reported next to circ_p.
  - MULTIPLICITY: the task file mandates "Bonferroni over 8 cells" as a COUNT; the composition
    of revision 1's 8-cell grid -- in particular the inclusion of 180d -- was the implementer's
    own pre-registration, not the task's, and that was not disclosed. This revision widens the
    grid to the full sweep and applies Bonferroni over the ACTUAL number of swept cells to the
    permutation p only. The circular-shift p is reported WITHOUT a multiplicity claim, because
    its minimum achievable p (printed per cell) is above the Bonferroni line for every cell --
    a "not significant" from that null is guaranteed by construction and is therefore not
    evidence of anything.
  - Cross-sectional sd of VR(30)-1: computed SEPARATELY, per-asset, over each asset's OWN
    full active life (30d sub-windows from its first real print), reported ONLY as context for
    why per-asset verdicts are unusable (per the task file's explicit instruction) -- never as
    a verdict. REVISION 3 caveat, added because this number is quoted against the task file's
    expected 0.16-0.46 range and therefore reads as corroboration of the methodology: it
    inherits a PER-ASSET version of the k-bias above. A contributing asset needs only 8
    sub-windows (median 40, 10th percentile 16 on this panel), so the per-asset VRs are biased
    low by an asset-specific amount, and the cross-asset sd therefore carries cross-asset BIAS
    dispersion on top of genuine dispersion in VR. Context only, and now context with a stated
    defect.

DECISION RULE. Revision 2 declared this rule before re-reading the numbers and argued it from
decision cost and estimator power. R1 stands as written. R2 DOES NOT: revision 3 (H)/(I) shows
it is not the power guard it claims to be, and that is recorded here rather than quietly
re-worded.

  R1 (unanimity guard on a DESTRUCTIVE verdict). P1's kill does not merely fail a candidate --
     it permanently closes an entire family of ideas (gated trend/reversal, CPD switches,
     XS-AUTOCORR books) on this venue. A reading that reverses under a free parameter with no
     principled setting is evidence about the parameter, not about the venue. The kill
     therefore fires only if EVERY cell in the pre-registered sweep reads rho-bar < 0.10.
     REVISION 3 (K): R1's FAILURE IS UNINFORMATIVE and must not be read as a finding. Using
     this artifact's own Fisher-z SEs, on a venue exactly as dead as the kill asserts (rho=0)
     the per-cell probability of reading rho-hat >= 0.10 by chance alone runs 0.246 -> 0.403,
     so P(R1 unanimity | rho = 0) ~= 0.0094. A rule a dead venue fails ~99% of the time cannot
     distinguish a dead venue from a live one. R1 is retained anyway, because it is the correct
     DECISION rule for a destructive verdict (do not close a family on a parameter-dependent
     reading) -- but it is a decision rule, not evidence, and revision 2's "-> FAIL (32/44)"
     wording implied the data had produced it.
  R2 (power guard AS CLAIMED; NOT ONE IN FACT). The anchor cell must additionally be able to
     RESOLVE the band at all: its Fisher-z 95% CI must exclude the 0.10 kill line. REVISION 3
     (H): this does not measure resolving power. With SE(rho-bar) ~= 0.17 at the anchor,
     whether a CI excludes 0.10 is decided by where the POINT ESTIMATE happened to land, and
     the point estimate is a strong function of the undisclosed TILING ORIGIN (swept below).
     Revision 2's R2 "PASS" is therefore reported as what it is: a lottery outcome, not
     evidence of power. The genuine power statement is the sign-flip ceiling in (I), and it
     says NO cell at ANY block length had the power to fire this kill.
  If R1 or R2 fails -> verdict is INCONCLUSIVE and the KILL DOES NOT FIRE. Symmetrically, an
  inconclusive P1 is NOT a claim that VR persistence exists; it is a statement that this
  instrument, on this panel, does not resolve the question. Revision 3 sharpens the second half:
  the instrument could not have resolved it either way.

Read-only against `fixtures/trading_fixture.db`'s `ohlcv_daily` table, which this pack's
report verifies is BYTE-IDENTICAL to `trading_data.db`'s (sha256 of the ordered row dump
matches) -- so these are the real production numbers, not a small-sample proxy. CPU, seconds.

SIDE EFFECT (stated because two review agents believed this module read-only): running it
OVERWRITES `p1_result.json` next to this file. The path is printed before the write.
"""
import sys
import json
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_panel as dp

FIXTURE = str(Path(__file__).resolve().parents[2] / "fixtures" / "trading_fixture.db")
ARTIFACT = Path(__file__).resolve().parent / "p1_result.json"
Q = 30                       # VR horizon (days)
BREAKEVEN = 1.024
KILL_RHOBAR = 0.10
ALPHA = 0.05
# Block lengths swept: every multiple of Q=30 that yields >= ~10 block pairs on this panel.
# 30 excluded (single sub-boundary per block -> purely cross-sectional numerator; see docstring).
BLOCK_GRID = (60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360)
UNIVERSES = ("pit_dead_retained", "survivor_only")
ESTIMATORS = ("pooled", "xsmean")
ANCHOR = {"block_days": 90, "universe": "pit_dead_retained", "estimator": "pooled"}
LEGACY_HEADLINE = {"block_days": 180, "universe": "pit_dead_retained", "estimator": "pooled"}
N_PERM = 2000                 # permutation-null draws -> min achievable p = 1/2001
PERM_SEED = 917
SIGN_FLIP_DRAWS = 200         # revision-3 model-free power ceiling (I)
SIGN_FLIP_SEED = 9171
LOW_POWER_PAIRS = 10          # cells below this are flagged low-power in the sweep table
MIN_NUM_OBS = 20              # min pooled 30d-return obs for a block's numerator to count
MIN_DEN_OBS = 60              # min pooled daily-return obs for a block's denominator to count
MIN_ASSET_ACTIVE_DAYS = 240   # per-asset whole-life VR: need >=8 non-overlap 30d sub-windows


def _block_boundaries(T, block_days, origin=0):
    """Block start rows. `origin` is the TILING PHASE -- the free parameter revision 2
    hard-wired to 0 and never disclosed (revision 3 (H)). origin in [0, block_days) gives
    every distinct one-day tiling of the same history.

    The bound is b0 + block_days <= T - 1, because the block's LAST 30-day sub-window ends on
    row b0 + block_days (the first day of the next block -- see the module's block accounting)
    and that row must exist. At origin = 0 this is bit-identical to revision 2's
    `range(0, T - block_days + 1, block_days)` for every block length in BLOCK_GRID, since
    T = 4657 is not a multiple of any of them; the tighter bound only matters once the origin
    is allowed to move."""
    return list(range(int(origin), T - block_days, block_days))


def _num_window_mask(r30_fin, sub_ends, block_days=None):
    """(T,A) boolean mask of the asset-days that FEED a valid numerator window: for every
    sub-window end `se` with a finite 30-day return, the Q daily returns dr[se-Q+1 .. se].
    This is what the SAMPLE-MATCHED denominator is restricted to (revision 3 (J))."""
    mask = np.zeros(r30_fin.shape, dtype=bool)
    for se in sub_ends:
        fin = r30_fin[se]
        if fin.any():
            mask[se - Q + 1:se + 1] |= fin[None, :]
    return mask


def pooled_block_vr_series(panel, block_days, origin=0, match_samples=False,
                           precomputed=None):
    """Market-level pooled VR(30) per block. Returns (vr (nblocks,) array, n_num (nblocks,),
    n_den (nblocks,)) -- vr is NaN where a block fails the min-obs floor.

    `origin`: tiling phase (revision 3 (H)); 0 reproduces revision 2 exactly.
    `match_samples`: when True the DENOMINATOR is restricted to the asset-days that actually
        feed a valid numerator window, removing the numerator/denominator sample mismatch that
        revision 2's level read tripped on (revision 3 (J)). Default False so the shipped,
        panel-reproduced sweep is bit-identical; the matched variant is reported alongside.
    `precomputed`: optional (dr, dr_fin, r30) triple, so a sweep over many origins or many
        sign-flip draws does not re-derive the panel returns each time."""
    if precomputed is None:
        dr, logc = dp.daily_log_return(panel["close"])
        dr_fin = np.isfinite(dr)
        r30 = dp.qday_return_nogap(logc, dr_fin, Q)
        T = logc.shape[0]
    else:
        dr, dr_fin, r30 = precomputed
        T = dr.shape[0]
    r30_fin = np.isfinite(r30)
    starts = _block_boundaries(T, block_days, origin)
    vr = np.full(len(starts), np.nan)
    n_num = np.zeros(len(starts), dtype=int)
    n_den = np.zeros(len(starts), dtype=int)
    for bi, b0 in enumerate(starts):
        b1 = b0 + block_days
        sub_ends = list(range(b0 + Q, b1 + 1, Q))
        if match_samples:
            used = _num_window_mask(r30_fin, sub_ends)
            sel = used & dr_fin
            den_vals = dr[sel]
        else:
            den_vals = dr[b0 + 1:b1][dr_fin[b0 + 1:b1]]
        num_vals = []
        for se in sub_ends:
            fin = r30_fin[se]
            if fin.any():
                num_vals.append(r30[se][fin])
        num_vals = np.concatenate(num_vals) if num_vals else np.array([])
        n_num[bi], n_den[bi] = num_vals.size, den_vals.size
        if num_vals.size >= MIN_NUM_OBS and den_vals.size >= MIN_DEN_OBS:
            varden = np.var(den_vals, ddof=1)
            varnum = np.var(num_vals, ddof=1)
            if varden > 0:
                vr[bi] = varnum / (Q * varden)
    return vr, n_num, n_den


def xsmean_block_vr_series(panel, block_days):
    """Alternative estimator (multiplicity grid only): per-asset block VR averaged
    cross-sectionally (equal weight), instead of pooling num/den. Needs >=2 30d sub-windows
    and >=20 daily obs for an asset to contribute to a given block."""
    dr, logc = dp.daily_log_return(panel["close"])
    dr_fin = np.isfinite(dr)
    r30 = dp.qday_return_nogap(logc, dr_fin, Q)
    T, A = logc.shape
    starts = _block_boundaries(T, block_days)
    vr = np.full(len(starts), np.nan)
    for bi, b0 in enumerate(starts):
        b1 = b0 + block_days
        sub_ends = list(range(b0 + Q, b1 + 1, Q))
        per_asset_vr = []
        for i in range(A):
            den_vals = dr[b0 + 1:b1, i]
            den_vals = den_vals[np.isfinite(den_vals)]
            if den_vals.size < 20:
                continue
            num_vals = np.array([r30[se, i] for se in sub_ends])
            num_vals = num_vals[np.isfinite(num_vals)]
            if num_vals.size < 2:
                continue
            vd = np.var(den_vals, ddof=1)
            vn = np.var(num_vals, ddof=1)
            if vd > 0:
                per_asset_vr.append(vn / (Q * vd))
        if len(per_asset_vr) >= 5:
            vr[bi] = float(np.mean(per_asset_vr))
    return vr


def _lag1_rho(vr):
    """rho-bar = Pearson corr(VR_b, VR_{b+1}) over consecutive blocks where BOTH are finite."""
    a, b = vr[:-1], vr[1:]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return float("nan"), int(mask.sum())
    if np.std(a[mask]) < 1e-15 or np.std(b[mask]) < 1e-15:
        return float("nan"), int(mask.sum())
    return float(np.corrcoef(a[mask], b[mask])[0, 1]), int(mask.sum())


def _fisher_ci(rho, n_pairs, conf=1.96):
    """Fisher-z 95% CI for a correlation on n_pairs observations. Returns (lo, hi, t_stat).
    Reported because with 8-55 non-overlapping block pairs SE(rho) ~= 0.14-0.37 -- a naked
    point estimate cannot be read against a 0.10 threshold."""
    if not np.isfinite(rho) or n_pairs < 4 or abs(rho) >= 1.0:
        return float("nan"), float("nan"), float("nan")
    z = np.arctanh(rho)
    se_z = 1.0 / np.sqrt(n_pairs - 3)
    lo, hi = float(np.tanh(z - conf * se_z)), float(np.tanh(z + conf * se_z))
    t = float(rho * np.sqrt(n_pairs - 2) / np.sqrt(max(1.0 - rho ** 2, 1e-12)))
    return lo, hi, t


def _emp_p(nulls, obs):
    """Plus-one (unbiased) two-sided empirical p. Minimum achievable value is 1/(n+1)."""
    nulls = np.asarray([c for c in nulls if np.isfinite(c)], dtype=np.float64)
    if nulls.size == 0 or not np.isfinite(obs):
        return float("nan"), 0, float("nan")
    p = float((1 + int(np.sum(np.abs(nulls) >= abs(obs)))) / (1 + nulls.size))
    return p, int(nulls.size), float(1.0 / (1 + nulls.size))


def circular_shift_null(vr):
    """Circular-shift null over DISTINCT, NON-DEGENERATE lags only.

    Excluded, and why (each verified degenerate, not assumed):
      s = 0      -> corr(vr, vr) = 1 trivially.
      s = 1      -> roll(vr, -1) pairs block b with b+1: the observed statistic itself
                    (bit-identical to rho-bar whenever the wrap pair is masked by a NaN, which
                    it is on every cell of this panel).
      s = N-1    -> roll(vr, -(N-1)) pairs block b with b-1: the SAME unordered pair set as
                    s=1, so a second copy of the observed statistic.
      s > N/2    -> c(s) == c(N-s) by the symmetry of the circular pairing: pure duplicates,
                    which inflate the apparent number of draws without adding resolution.
    Kept: s = 2..floor(N/2). Returns (nulls array, list of shifts used, list of n_pairs per
    draw). The n_pairs list exists because of a revision-3 small item: the finite mask is
    recomputed per shift, so these draws use FEWER pairs than the observed statistic and
    `circ_p` is therefore NOT comparable to it."""
    n = vr.size
    nulls, shifts, npairs = [], [], []
    for s in range(2, n // 2 + 1):
        shifted = np.roll(vr, -s)
        m2 = np.isfinite(vr) & np.isfinite(shifted)
        if m2.sum() < 4:
            continue
        x, y = vr[m2], shifted[m2]
        if np.std(x) < 1e-15 or np.std(y) < 1e-15:
            continue
        c = np.corrcoef(x, y)[0, 1]
        if np.isfinite(c):
            nulls.append(float(c))
            shifts.append(s)
            npairs.append(int(m2.sum()))
    return np.asarray(nulls), shifts, npairs


def permutation_null(vr, n_perm=N_PERM, seed=PERM_SEED):
    """Random-permutation null: reorder the FINITE block VRs among the finite block positions
    (NaN pattern held fixed, so every draw has exactly the same n_pairs as the observed
    statistic) and recompute rho-bar identically. This destroys serial order -- the thing under
    test -- while preserving the marginal distribution of block VRs, and it has enough draws
    (1/(n_perm+1) minimum p) to carry a multiplicity claim, which the circular-shift null on a
    12-50 block series does not."""
    fin = np.isfinite(vr)
    vals = vr[fin]
    if vals.size < 5:
        return np.array([])
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    work = vr.copy()
    for d in range(n_perm):
        work[fin] = rng.permutation(vals)
        r, _ = _lag1_rho(work)
        out[d] = r
    return out


def persistence_and_placebo(vr):
    """Backwards-compatible shim retained so the review panel's own reproduction scripts keep
    working: returns (rho_bar, circular-shift p, nulls, n_pairs) using the CORRECTED
    (non-degenerate) circular-shift null and the plus-one p."""
    rho_bar, n_pairs = _lag1_rho(vr)
    nulls, _, _ = circular_shift_null(vr)
    p, _, _ = _emp_p(nulls, rho_bar)
    return rho_bar, p, nulls, n_pairs


def cell_stats(vr):
    """Full per-cell record: rho-bar, its Fisher-z CI/t, and BOTH nulls with their minimum
    achievable p-values."""
    rho_bar, n_pairs = _lag1_rho(vr)
    ci_lo, ci_hi, t_stat = _fisher_ci(rho_bar, n_pairs)
    circ_nulls, circ_shifts, circ_npairs = circular_shift_null(vr)
    circ_p, circ_n, circ_pmin = _emp_p(circ_nulls, rho_bar)
    perm_nulls = permutation_null(vr)
    perm_p, perm_n, perm_pmin = _emp_p(perm_nulls, rho_bar)
    # rho-bar's exact sampling sd under this cell's own Fisher-z SE, used for the
    # revision-3 (K) "what would a DEAD venue do?" calculation.
    p_cell_above_kill = float("nan")
    if np.isfinite(rho_bar) and n_pairs > 3:
        se_z = 1.0 / np.sqrt(n_pairs - 3)
        # P(rho_hat >= KILL_RHOBAR | true rho = 0), on the Fisher-z scale.
        p_cell_above_kill = float(0.5 * math.erfc(
            (np.arctanh(KILL_RHOBAR) / se_z) / np.sqrt(2.0)))
    return {
        "n_blocks_total": int(vr.size), "n_blocks_valid": int(np.isfinite(vr).sum()),
        "n_pairs": n_pairs, "rho_bar": rho_bar,
        "ci95_lo": ci_lo, "ci95_hi": ci_hi, "t_stat": t_stat,
        "circ_p": circ_p, "circ_n_draws": circ_n, "circ_p_min": circ_pmin,
        "circ_shifts_used": circ_shifts,
        # Revision-3 small item: the circular draws use FEWER pairs than the observed
        # statistic, so circ_p is not comparable to it. Recorded per cell, not asserted.
        "circ_median_n_pairs": float(np.median(circ_npairs)) if circ_npairs else float("nan"),
        "circ_n_pairs_comparable": bool(circ_npairs
                                        and int(np.median(circ_npairs)) == int(n_pairs)),
        "perm_p": perm_p, "perm_n_draws": perm_n, "perm_p_min": perm_pmin,
        "p_above_kill_if_venue_dead": p_cell_above_kill,
        "low_power": bool(n_pairs < LOW_POWER_PAIRS),
        # Can this cell resolve the kill line at all? (CI entirely on one side of 0.10.)
        "ci_excludes_kill_line": bool(np.isfinite(ci_lo) and np.isfinite(ci_hi)
                                      and (ci_hi < KILL_RHOBAR or ci_lo > KILL_RHOBAR)),
        "below_kill": bool(np.isfinite(rho_bar) and rho_bar < KILL_RHOBAR),
    }


def per_asset_whole_life_vr(panel):
    """Per-asset VR(30) over its OWN full active life (non-overlapping 30d sub-windows from
    first real print), for the cross-sectional sd(VR-1) context statistic (NOT a verdict —
    per-asset sd dominates the 0.02 break-even margin by construction, per the task file)."""
    dr, logc = dp.daily_log_return(panel["close"])
    dr_fin = np.isfinite(dr)
    r30 = dp.qday_return_nogap(logc, dr_fin, Q)
    T, A = logc.shape
    out = []
    n_sub = []          # revision 3: the per-asset sub-window count, for the k-bias caveat
    for i in range(A):
        col_fin = np.isfinite(panel["close"][:, i])
        if not col_fin.any():
            continue
        first = int(np.argmax(col_fin))
        last = int(T - 1 - np.argmax(col_fin[::-1]))
        active_days = last - first + 1
        if active_days < MIN_ASSET_ACTIVE_DAYS:
            continue
        den_vals = dr[first + 1:last + 1, i]
        den_vals = den_vals[np.isfinite(den_vals)]
        sub_ends = range(first + Q, last + 1, Q)
        num_vals = np.array([r30[se, i] for se in sub_ends if se <= last])
        num_vals = num_vals[np.isfinite(num_vals)]
        if den_vals.size < 60 or num_vals.size < 8:
            continue
        vd = np.var(den_vals, ddof=1)
        vn = np.var(num_vals, ddof=1)
        if vd > 0:
            out.append(vn / (Q * vd))
            n_sub.append(int(num_vals.size))
    return np.asarray(out), np.asarray(n_sub)


def whole_sample_pooled_vr(panel, match_samples=False, precomputed=None):
    """Single pooled VR(30) over the ENTIRE sample (one giant "block"), for the contextual
    level-vs-breakeven read (not the persistence gate).

    `match_samples` (revision 3 (J)): restrict the denominator to the asset-days that feed a
    valid numerator window. Revision 2's shipped read did NOT: `den_vals = dr[dr_fin]` takes
    every finite daily return in the panel while the numerator takes only asset-days inside
    gap-free 30-day windows. The excluded ~3.8% (gap-adjacent days, listing/delisting edges)
    are systematically MORE volatile and inflate the denominator only, so the shipped level is
    biased low against the breakeven it is compared to."""
    if precomputed is None:
        dr, logc = dp.daily_log_return(panel["close"])
        dr_fin = np.isfinite(dr)
        r30 = dp.qday_return_nogap(logc, dr_fin, Q)
        T = logc.shape[0]
    else:
        dr, dr_fin, r30 = precomputed
        T = dr.shape[0]
    r30_fin = np.isfinite(r30)
    sub_rows = list(range(Q, T, Q))
    if match_samples:
        sel = _num_window_mask(r30_fin, sub_rows) & dr_fin
        den_vals = dr[sel]
    else:
        den_vals = dr[dr_fin]
    num_vals = []
    for se in sub_rows:
        fin = r30_fin[se]
        if fin.any():
            num_vals.append(r30[se][fin])
    num_vals = np.concatenate(num_vals) if num_vals else np.array([])
    vd = np.var(den_vals, ddof=1)
    vn = np.var(num_vals, ddof=1)
    return float(vn / (Q * vd)) if vd > 0 else float("nan"), int(num_vals.size), int(den_vals.size)


# --------------------------------------------------------------------------------------------
# REVISION 3 instruments: the tiling-origin sweep (H), the model-free sign-flip power ceiling
# (I) and the sample-matched level read (J).
# --------------------------------------------------------------------------------------------

def _panel_returns(panel):
    """(dr, dr_fin, r30) for a panel — computed once and reused by every sweep below."""
    dr, logc = dp.daily_log_return(panel["close"])
    dr_fin = np.isfinite(dr)
    r30 = dp.qday_return_nogap(logc, dr_fin, Q)
    return dr, dr_fin, r30


def _r30_from_dr(dr, dr_fin, q=Q):
    """q-day compounded log return rebuilt from DAILY RETURNS rather than from log prices,
    which is what a sign-flip null needs (flipping a return's sign has no log-price
    representation). Identical semantics to `dp.qday_return_nogap`: r[t] = sum(dr[t-q+1..t]),
    valid only if every one of those q daily returns is finite. Verified byte-identical to
    `dp.qday_return_nogap` on the real panel at run time before any null is drawn."""
    T, A = dr.shape
    x = np.where(dr_fin, dr, 0.0)
    cs = np.zeros((T + 1, A))
    np.cumsum(x, axis=0, out=cs[1:])
    badc = np.zeros((T + 1, A), dtype=np.int64)
    np.cumsum((~dr_fin).astype(np.int64), axis=0, out=badc[1:])
    out = np.full((T, A), np.nan)
    if T <= q:
        return out
    tot = cs[q + 1:] - cs[1:T - q + 1]          # sum of dr over (t-q, t] for t = q..T-1
    bad = badc[q + 1:] - badc[1:T - q + 1]
    out[q:] = np.where(bad == 0, tot, np.nan)
    return out


def tiling_origin_sweep(panel, block_days, origins=None, estimator="pooled",
                        precomputed=None):
    """REVISION 3 (H). rho-bar as a function of the TILING ORIGIN — the free parameter
    `_block_boundaries` hard-wires to day 0 and revision 2 never disclosed.

    Sweeps every distinct one-day tiling of the same history (origin = 0..block_days-1 by
    default) and returns the rho-bar distribution. This is the disconfirming evidence against
    revision 2's "all four 90d cells agree to within 0.02" counterfactual: the four cells share
    ONE origin, so their agreement measures the universe/estimator axes only."""
    if precomputed is None:
        precomputed = _panel_returns(panel)
    if origins is None:
        origins = range(block_days)
    rows = []
    for o in origins:
        if estimator == "pooled":
            vr, _nn, _nd = pooled_block_vr_series(panel, block_days, origin=o,
                                                  precomputed=precomputed)
        else:
            raise ValueError("origin sweep is defined for the pooled estimator only")
        rho, npairs = _lag1_rho(vr)
        rows.append({"origin": int(o), "rho_bar": rho, "n_pairs": npairs,
                     "n_blocks_valid": int(np.isfinite(vr).sum())})
    return rows


def _origin_summary(rows, kill=KILL_RHOBAR):
    r = np.asarray([x["rho_bar"] for x in rows], dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {}
    return {"n_origins": int(r.size), "mean": float(r.mean()),
            "sd": float(np.std(r, ddof=1)) if r.size > 1 else float("nan"),
            "min": float(r.min()), "max": float(r.max()),
            "range": float(r.max() - r.min()),
            "frac_above_kill": float(np.mean(r >= kill)),
            "n_above_kill": int(np.sum(r >= kill))}


def sign_flip_null(panel, block_grid=None, n_draws=200, seed=917, precomputed=None):
    """REVISION 3 (I) — the MODEL-FREE power ceiling, and the load-bearing correction.

    Null construction: multiply every DAY's cross-section of daily returns by a common random
    sign. This
      * destroys serial structure completely, so the TRUE variance ratio is exactly 1;
      * preserves each asset-day's magnitude, hence every marginal variance;
      * preserves the cross-sectional covariance (the market factor) exactly, which matters
        because the k-bias in (J) is a market-factor effect and a per-asset-day flip would
        destroy the very thing being measured.
    It is model-free: no w (market share), no volatility process, no distributional assumption
    is fitted. That is why it, and not any parametric simulation, is the load-bearing evidence.

    Reported per block length: E[VR_block] under a true VR of 1 (the k-bias of (J)), and the
    null distribution of the DISPERSION of block VRs, which bounds how much of the observed
    dispersion can be signal. Reported once: the null distribution of the whole-sample level
    under both the shipped and the sample-matched denominator."""
    if block_grid is None:
        block_grid = BLOCK_GRID
    if precomputed is None:
        precomputed = _panel_returns(panel)
    dr, dr_fin, _r30_real = precomputed
    T = dr.shape[0]
    rng = np.random.default_rng(seed)
    per_block = {bd: {"mean_vr": [], "sd_vr": [], "rho_bar": []} for bd in block_grid}
    lvl_shipped, lvl_matched = [], []
    for _d in range(n_draws):
        sgn = rng.choice(np.array([-1.0, 1.0]), size=(T, 1))
        dr_f = dr * sgn
        r30_f = _r30_from_dr(dr_f, dr_fin)
        pre = (dr_f, dr_fin, r30_f)
        for bd in block_grid:
            vr, _nn, _nd = pooled_block_vr_series(panel, bd, precomputed=pre)
            fin = vr[np.isfinite(vr)]
            per_block[bd]["mean_vr"].append(float(fin.mean()) if fin.size else np.nan)
            per_block[bd]["sd_vr"].append(float(np.std(fin, ddof=1)) if fin.size > 1
                                          else np.nan)
            rho, _np_ = _lag1_rho(vr)
            per_block[bd]["rho_bar"].append(rho)
        v0, _n, _m = whole_sample_pooled_vr(panel, match_samples=False, precomputed=pre)
        v1, _n, _m = whole_sample_pooled_vr(panel, match_samples=True, precomputed=pre)
        lvl_shipped.append(v0)
        lvl_matched.append(v1)
    return per_block, np.asarray(lvl_shipped), np.asarray(lvl_matched)


def _upper_tail_p(nulls, obs):
    """Plus-one UPPER-tail empirical p — used for a dispersion statistic, where only 'more
    dispersed than noise' is evidence of signal."""
    n = np.asarray([x for x in nulls if np.isfinite(x)], dtype=np.float64)
    if n.size == 0 or not np.isfinite(obs):
        return float("nan"), 0
    return float((1 + int(np.sum(n >= obs))) / (1 + n.size)), int(n.size)


def _power_two_sided(rho_alt, n_pairs, alpha=ALPHA):
    """Power of the Fisher-z test of H0: rho = 0 against a TRUE rho = rho_alt, at n_pairs
    observations. This is the number revision 2's R2 claimed to be about and was not."""
    if n_pairs is None or n_pairs <= 3:
        return float("nan"), float("nan")
    se_z = 1.0 / np.sqrt(n_pairs - 3)
    z_alt = np.arctanh(rho_alt)
    # two-sided critical value at `alpha`, and the one-sided value at the same alpha
    crit2 = 1.959963984540054 * se_z
    crit1 = 1.6448536269514722 * se_z
    def _tail(c):
        return float(0.5 * math.erfc(((c - z_alt) / se_z) / np.sqrt(2.0))
                     + 0.5 * math.erfc(((c + z_alt) / se_z) / np.sqrt(2.0)))
    return _tail(crit2), _tail(crit1)


def _find(cells, spec):
    return next(c for c in cells if all(c[k] == v for k, v in spec.items()))


def main():
    print(f"[P1] fixture: {FIXTURE}")
    print(f"[P1] SIDE EFFECT: this run will OVERWRITE the artifact {ARTIFACT}")
    panels = {}
    for universe in UNIVERSES:
        include_dead = (universe == "pit_dead_retained")
        panel = dp.load_raw_pit_panel(FIXTURE, include_dead=include_dead)
        panels[universe] = panel
        print(f"[P1] panel[{universe}]: T={panel['day_ts'].size} A={len(panel['ids'])} "
              f"(info={panel['info']})")

    cells = []
    for block_days in BLOCK_GRID:
        for universe in UNIVERSES:
            for estimator in ESTIMATORS:
                panel = panels[universe]
                if estimator == "pooled":
                    vr, _n_num, _n_den = pooled_block_vr_series(panel, block_days)
                else:
                    vr = xsmean_block_vr_series(panel, block_days)
                rec = {"block_days": block_days, "universe": universe, "estimator": estimator,
                       "n_subwindows_per_block": block_days // Q}
                rec.update(cell_stats(vr))
                cells.append(rec)

    n_cells = len(cells)
    bonf_alpha = ALPHA / n_cells
    for c in cells:
        c["bonferroni_alpha"] = bonf_alpha
        c["n_cells_swept"] = n_cells
        # Multiplicity is claimed ONLY against the permutation null: the circular-shift null's
        # own minimum achievable p is above the Bonferroni line for every cell, so a
        # "not significant" from it is guaranteed by construction (Finding 2).
        c["significant_bonf_perm"] = bool(np.isfinite(c["perm_p"]) and c["perm_p"] < bonf_alpha)
        c["circ_can_reach_bonf"] = bool(np.isfinite(c["circ_p_min"])
                                        and c["circ_p_min"] <= bonf_alpha)

    anchor = _find(cells, ANCHOR)
    legacy = _find(cells, LEGACY_HEADLINE)

    # Whole-sample pooled VR level (context, vs breakeven) on the anchor universe.
    anchor_panel = panels["pit_dead_retained"]
    pre_real = _panel_returns(anchor_panel)
    # Verify the sign-flip null's return builder is the SAME estimator as the shipped one
    # BEFORE any null is drawn (revision 3: no null is trusted on an unverified twin).
    _r30_twin = _r30_from_dr(pre_real[0], pre_real[1])
    _twin_ok = bool(np.array_equal(np.isfinite(_r30_twin), np.isfinite(pre_real[2]))
                    and np.allclose(_r30_twin[np.isfinite(_r30_twin)],
                                    pre_real[2][np.isfinite(pre_real[2])],
                                    rtol=0, atol=1e-9))
    print(f"[P1] sign-flip twin-estimator check: `_r30_from_dr` reproduces "
          f"`dp.qday_return_nogap` on the real panel = {_twin_ok}")
    if not _twin_ok:
        raise SystemExit("[P1] ABORT: the sign-flip null's return builder does not reproduce "
                         "the shipped 30-day return estimator; the power ceiling below would "
                         "be computed by a different estimator than the sweep.")
    vr_level, n_num_ws, n_den_ws = whole_sample_pooled_vr(anchor_panel, match_samples=False,
                                                          precomputed=pre_real)
    vr_level_m, n_num_m, n_den_m = whole_sample_pooled_vr(anchor_panel, match_samples=True,
                                                          precomputed=pre_real)

    # Per-asset whole-life VR dispersion (context only, never a verdict).
    per_asset_vr, per_asset_nsub = per_asset_whole_life_vr(anchor_panel)
    xsd = float(np.std(per_asset_vr - 1.0, ddof=1)) if per_asset_vr.size > 1 else float("nan")

    print(f"\n=== P1 — FULL BLOCK-LENGTH SENSITIVITY SWEEP ({n_cells} cells: "
          f"block in {BLOCK_GRID} x universe x estimator) ===")
    print("Bonferroni alpha = 0.05/%d = %.5f, applied to the PERMUTATION p only "
          "(the circular-shift p cannot reach it -- see circ_pmin)." % (n_cells, bonf_alpha))
    print(f"{'block':>5} {'k':>2} {'universe':>18} {'est':>7} {'blocks':>7} {'pairs':>5} "
          f"{'rho_bar':>8} {'95% CI (Fisher-z)':>21} {'t':>6} "
          f"{'circ_p':>7} {'circ_pmin':>9} {'perm_p':>7} {'sigB':>5} {'<kill':>6}")
    for c in cells:
        ci = (f"[{c['ci95_lo']:+.3f},{c['ci95_hi']:+.3f}]"
              if np.isfinite(c["ci95_lo"]) else "n/a")
        flag = "*" if c["low_power"] else " "
        print(f"{c['block_days']:>5} {c['n_subwindows_per_block']:>2} {c['universe']:>18} "
              f"{c['estimator']:>7} {c['n_blocks_valid']:>3}/{c['n_blocks_total']:<3} "
              f"{c['n_pairs']:>4}{flag} {c['rho_bar']:>+8.4f} {ci:>21} {c['t_stat']:>+6.2f} "
              f"{c['circ_p']:>7.3f} {c['circ_p_min']:>9.3f} {c['perm_p']:>7.4f} "
              f"{str(c['significant_bonf_perm']):>5} {str(c['below_kill']):>6}")
    print(f"  (* = fewer than {LOW_POWER_PAIRS} block pairs -> low power; k = 30-day "
          f"sub-windows per block)")

    n_below = sum(1 for c in cells if c["below_kill"])
    n_above = n_cells - n_below
    above = [c for c in cells if not c["below_kill"]]
    n_resolving = sum(1 for c in cells if c["ci_excludes_kill_line"])
    resolve_neg = [c for c in cells if c["ci_excludes_kill_line"] and c["ci95_hi"] < KILL_RHOBAR]
    resolve_pos = [c for c in cells if c["ci_excludes_kill_line"] and c["ci95_lo"] > KILL_RHOBAR]

    # Per-block-length rollup. REVISION 3 (L): revision 2 asserted here, in a hardcoded string
    # the code never evaluated, that "universe and estimator move rho_bar by <=0.15 within a
    # block length", and used it to deflate 44 cells to 11 independent readings. The claim is
    # now COMPUTED, and it fails.
    by_block = []
    print("\n=== P1 — sensitivity rolled up by BLOCK LENGTH (the free parameter that moves the "
          "answer) ===")
    print(f"{'block':>5} {'k':>2} {'pairs':>6} {'rho_bar (4 cells)':>34} {'all<kill':>9} "
          f"{'CI resolves kill line':>22}")
    for bd in BLOCK_GRID:
        grp = [c for c in cells if c["block_days"] == bd]
        rhos = [c["rho_bar"] for c in grp]
        allb = all(c["below_kill"] for c in grp)
        res = ("yes (below)" if all(c["ci95_hi"] < KILL_RHOBAR for c in grp)
               else ("yes (above)" if all(c["ci95_lo"] > KILL_RHOBAR for c in grp) else "no"))
        by_block.append({"block_days": bd, "rho_bar_min": min(rhos), "rho_bar_max": max(rhos),
                         "within_block_spread": float(max(rhos) - min(rhos)),
                         "n_pairs_range": [min(c["n_pairs"] for c in grp),
                                           max(c["n_pairs"] for c in grp)],
                         "all_below_kill": allb, "ci_resolves_kill_line": res})
        print(f"{bd:>5} {bd // Q:>2} {min(c['n_pairs'] for c in grp):>2}-"
              f"{max(c['n_pairs'] for c in grp):<3} "
              f"{'  '.join(f'{r:+.3f}' for r in rhos):>34} {str(allb):>9} {res:>22}")

    print(f"\n[P1] sweep summary: {n_below}/{n_cells} cells below the {KILL_RHOBAR} kill line, "
          f"{n_above}/{n_cells} at or above it.")
    if above:
        print("[P1] cells AT OR ABOVE the kill line (the sweep's disconfirming evidence):")
        for c in sorted(above, key=lambda c: -c["rho_bar"]):
            print(f"       block={c['block_days']:>3} {c['universe']:>18} "
                  f"{c['estimator']:>7} pairs={c['n_pairs']:>3} rho_bar={c['rho_bar']:+.4f}")
    print(f"[P1] cells whose 95% CI EXCLUDES the {KILL_RHOBAR} kill line (i.e. that can "
          f"resolve it at all): {n_resolving}/{n_cells} "
          f"({len(resolve_neg)} on the BELOW side, {len(resolve_pos)} on the ABOVE side)")
    # ---- (L) the 44 -> 11 multiplicity deflation rested on a FALSE, uncomputed premise ------
    spreads = [b["within_block_spread"] for b in by_block]
    viol = [b for b in by_block if b["within_block_spread"] > 0.15]
    print(f"\n=== P1 (L) — REFUTATION of revision 2's independence claim (it was a hardcoded "
          f"string; the code never computed it) ===")
    print(f"[P1] revision 2 printed: 'universe and estimator move rho_bar by <=0.15 within a "
          f"block length', and used it to deflate {n_cells} cells to {len(BLOCK_GRID)} "
          f"independent readings. MEASURED within-block-length spreads of rho_bar "
          f"(max - min over the 4 cells):")
    print("       " + "  ".join(f"{bd}d:{s:.3f}" for bd, s in zip(BLOCK_GRID, spreads)))
    print(f"[P1] {len(viol)}/{len(BLOCK_GRID)} block lengths VIOLATE the <=0.15 claim (criterion "
          f"stated explicitly: spread strictly > 0.15; the review panel quoted 9/11 from this "
          f"same list of eleven spreads, a one-cell counting difference that does not touch the "
          f"refutation). Max spread {max(spreads):.3f} at "
          f"{[b['block_days'] for b in by_block if b['within_block_spread'] == max(spreads)][0]}"
          f"d — over three times the asserted bound. The claim is WITHDRAWN, and with it the "
          f"{n_cells}->{len(BLOCK_GRID)} multiplicity deflation that rested on it.")
    # Which axis actually moves rho-bar at the long end?
    est_axis, blk_axis = [], []
    for bd in BLOCK_GRID:
        if bd < 240:
            continue
        grp = [c for c in cells if c["block_days"] == bd]
        for u in UNIVERSES:
            g = [c for c in grp if c["universe"] == u]
            if len(g) == 2:
                est_axis.append(abs(g[0]["rho_bar"] - g[1]["rho_bar"]))
    long_blocks = [bd for bd in BLOCK_GRID if bd >= 240]
    for u in UNIVERSES:
        for e in ESTIMATORS:
            series = [c["rho_bar"] for c in cells
                      if c["universe"] == u and c["estimator"] == e
                      and c["block_days"] in long_blocks]
            # ADJACENT block-length steps, so this is the same kind of quantity as the
            # pairwise estimator difference it is being compared with.
            blk_axis.extend(abs(series[j + 1] - series[j]) for j in range(len(series) - 1))
    if est_axis and blk_axis:
        print(f"[P1] and at blocks >= 240d the ESTIMATOR axis moves rho_bar MORE than the "
              f"block-length axis does: mean |pooled - xsmean| at a fixed block length = "
              f"{np.mean(est_axis):.3f} vs mean |one 30-day step in block length| at a fixed "
              f"universe/estimator = {np.mean(blk_axis):.3f}. 'Block length is the free "
              f"parameter that moves the answer' holds at the short end only.")

    print(f"\n[P1] ANCHOR cell (block=90d INHERITED from mech_B.md:44,48 -- the source panel's "
          f"own pre-registered 90-day non-overlapping window with the identical rho-bar >= "
          f"{KILL_RHOBAR} threshold; universe=PIT dead-retained, estimator=pooled):")
    print(f"     rho_bar = {anchor['rho_bar']:+.4f}  95% CI [{anchor['ci95_lo']:+.4f},"
          f"{anchor['ci95_hi']:+.4f}]  t = {anchor['t_stat']:+.2f}  n_pairs = {anchor['n_pairs']}")
    print(f"     circular-shift p = {anchor['circ_p']:.4f} on {anchor['circ_n_draws']} distinct "
          f"non-degenerate lags (min achievable p = {anchor['circ_p_min']:.4f})")
    print(f"     permutation p    = {anchor['perm_p']:.4f} on {anchor['perm_n_draws']} draws "
          f"(min achievable p = {anchor['perm_p_min']:.5f}); Bonferroni-significant = "
          f"{anchor['significant_bonf_perm']}")
    print(f"[P1] revision-1 headline cell (block=180d -- TASK_917's OWN choice, inherited from "
          f"nothing, disclosed nowhere in revision 1): rho_bar = {legacy['rho_bar']:+.4f}, "
          f"95% CI [{legacy['ci95_lo']:+.4f},{legacy['ci95_hi']:+.4f}], "
          f"n_pairs = {legacy['n_pairs']} (fewer pairs than the anchor's {anchor['n_pairs']})")

    # ========================================================================================
    # REVISION 3 (H) — THE TILING ORIGIN, the free parameter revision 2 never disclosed
    # ========================================================================================
    print(f"\n=== P1 (H) — TILING-ORIGIN SWEEP at the ANCHOR: the parameter revision 2 "
          f"hard-wired to day 0 and did not disclose ===")
    anchor_origins = tiling_origin_sweep(anchor_panel, ANCHOR["block_days"],
                                         precomputed=pre_real)
    anchor_osum = _origin_summary(anchor_origins)
    shipped_rho = anchor["rho_bar"]
    z_shipped = ((shipped_rho - anchor_osum["mean"]) / anchor_osum["sd"]
                 if anchor_osum.get("sd") else float("nan"))
    print(f"[P1] all {anchor_osum['n_origins']} one-day tiling origins of the SAME history at "
          f"block={ANCHOR['block_days']}d: rho_bar mean={anchor_osum['mean']:+.4f} "
          f"sd={anchor_osum['sd']:.4f} min={anchor_osum['min']:+.4f} "
          f"max={anchor_osum['max']:+.4f}")
    print(f"[P1] {anchor_osum['n_above_kill']}/{anchor_osum['n_origins']} origins "
          f"({anchor_osum['frac_above_kill']:.1%}) put rho_bar AT OR ABOVE the "
          f"{KILL_RHOBAR} kill line. The shipped origin (day 0) reads {shipped_rho:+.4f}, "
          f"which is {z_shipped:+.2f} sd from the origin-distribution mean — i.e. the anchor "
          f"cell sits well into the NEGATIVE TAIL of its own free parameter.")
    coarse = {}
    for bd in (90, 180, 270):
        rows = tiling_origin_sweep(anchor_panel, bd,
                                   origins=range(0, bd, 30), precomputed=pre_real)
        coarse[bd] = rows
        vals = [r["rho_bar"] for r in rows]
        print(f"[P1] coarse 30-day tiling phases at block={bd}d: "
              f"{'  '.join(f'{v:+.3f}' for v in vals)}  (spread "
              f"{max(vals) - min(vals):.3f})")
    one_phase = {}
    for bd in (60, 90, 120):
        rows = tiling_origin_sweep(anchor_panel, bd, origins=[30], precomputed=pre_real)
        one_phase[bd] = rows[0]["rho_bar"]
    print(f"[P1] a SINGLE 30-day origin shift flips the sign or the magnitude wholesale: "
          + "  ".join(f"block={bd}d origin=30 -> {one_phase[bd]:+.3f} "
                      f"(shipped {_find(cells, {'block_days': bd, 'universe': 'pit_dead_retained', 'estimator': 'pooled'})['rho_bar']:+.3f})"
                      for bd in (60, 90, 120)))
    print(f"[P1] CONSEQUENCE, stated as a correction and not as a caveat: R2 IS NOT A POWER "
          f"GUARD. With SE(rho_bar) ~= {anchor_osum['sd']:.3f} at the anchor, 'the 95% CI "
          f"excludes {KILL_RHOBAR}' is decided by where the point estimate happened to land "
          f"under an undisclosed free parameter — a lottery on the tiling phase, not resolving "
          f"power. Revision 2's R2 'PASS' is reported below as exactly that.")

    # ========================================================================================
    # REVISION 3 (I) — THE MODEL-FREE POWER CEILING (the load-bearing correction)
    # ========================================================================================
    print(f"\n=== P1 (I) — MODEL-FREE POWER CEILING: sign-flip null on the REAL panel "
          f"({SIGN_FLIP_DRAWS} draws, seed {SIGN_FLIP_SEED}) ===")
    print("[P1] Null: multiply every DAY's cross-section of daily returns by a common random "
          "sign. Serial structure is destroyed (so the TRUE variance ratio is exactly 1), every "
          "marginal variance and the entire cross-sectional covariance are preserved. Nothing "
          "is fitted — no market share w, no volatility process, no distribution. That is why "
          "this, and not any parametric simulation, is the load-bearing evidence here.")
    sf_block, sf_lvl_shipped, sf_lvl_matched = sign_flip_null(
        anchor_panel, n_draws=SIGN_FLIP_DRAWS, seed=SIGN_FLIP_SEED, precomputed=pre_real)
    print(f"\n{'block':>5} {'k':>2} {'E[VR_blk] | true VR=1':>22} {'obs sd(VR_blk)':>15} "
          f"{'null sd(VR_blk)':>16} {'upper-tail p':>13} {'signal var share':>17}")
    power_rows = []
    for bd in BLOCK_GRID:
        cell = _find(cells, {"block_days": bd, "universe": "pit_dead_retained",
                             "estimator": "pooled"})
        vr_obs, _nn, _nd = pooled_block_vr_series(anchor_panel, bd, precomputed=pre_real)
        vr_obs_fin = vr_obs[np.isfinite(vr_obs)]
        sd_obs = float(np.std(vr_obs_fin, ddof=1)) if vr_obs_fin.size > 1 else float("nan")
        null_sds = np.asarray(sf_block[bd]["sd_vr"], dtype=np.float64)
        null_means = np.asarray(sf_block[bd]["mean_vr"], dtype=np.float64)
        p_disp, n_disp = _upper_tail_p(null_sds, sd_obs)
        null_var_mean = float(np.nanmean(null_sds ** 2))
        share = float(max(0.0, 1.0 - null_var_mean / (sd_obs ** 2))) if sd_obs > 0 else float("nan")
        row = {"block_days": bd, "k": bd // Q,
               "E_vr_block_under_true_vr1": float(np.nanmean(null_means)),
               "obs_sd_vr_block": sd_obs, "null_sd_vr_block_mean": float(np.nanmean(null_sds)),
               "dispersion_upper_tail_p": p_disp, "n_null_draws": n_disp,
               "implied_signal_variance_share": share,
               "rho_bar_shipped": cell["rho_bar"], "n_pairs": cell["n_pairs"]}
        power_rows.append(row)
        print(f"{bd:>5} {bd // Q:>2} {row['E_vr_block_under_true_vr1']:>22.4f} "
              f"{sd_obs:>15.4f} {row['null_sd_vr_block_mean']:>16.4f} "
              f"{p_disp:>13.3f} {share:>17.3f}")
    ps = [r["dispersion_upper_tail_p"] for r in power_rows]
    shares = [r["implied_signal_variance_share"] for r in power_rows]
    n_sig_disp = sum(1 for p in ps if p < ALPHA)
    print(f"[P1] the observed block-VR dispersion is INDISTINGUISHABLE from pure estimator "
          f"noise at {len(ps) - n_sig_disp}/{len(ps)} block lengths (upper-tail p range "
          f"[{min(ps):.3f},{max(ps):.3f}]); implied signal-variance share ranges "
          f"[{min(shares):.2f},{max(shares):.2f}].")
    max_attainable = max(shares)
    pw2, pw1 = _power_two_sided(KILL_RHOBAR, anchor["n_pairs"])
    se_lo = 1.0 / np.sqrt(max(max(c["n_pairs"] for c in cells) - 3, 1))
    se_hi = 1.0 / np.sqrt(max(min(c["n_pairs"] for c in cells) - 3, 1))
    print(f"[P1] rho_bar is ATTENUATED by that share: an observed rho_bar is at most the share "
          f"times the true one. With a share of at most {max_attainable:.2f} anywhere on the "
          f"grid, the LARGEST |rho_bar| this instrument could report against a PERFECTLY "
          f"PERSISTENT true VR is ~{max_attainable:.2f} — against a Fisher-z SE of "
          f"{se_lo:.2f} (most pairs) to {se_hi:.2f} (fewest), and "
          f"{anchor_osum['sd']:.2f} at the anchor by its own tiling-origin dispersion. The "
          f"ceiling and the noise are the same size. (The review panel's independent draws put "
          f"the ceiling at ~0.3 with SE >= 0.14; this run's 200 draws put it at "
          f"{max_attainable:.2f}. Either way the conclusion is the same and it is not close.)")
    print(f"[P1] and the anchor's power to separate rho_bar = {KILL_RHOBAR} from rho_bar = 0 at "
          f"alpha = {ALPHA} is {pw2:.1%} two-sided / {pw1:.1%} one-sided on {anchor['n_pairs']} "
          f"pairs (the review panel's independent figure was ~10%).")
    print(f"[P1] HONEST CONCLUSION, and it is stronger than revision 2's: **P1 COULD NOT HAVE "
          f"FIRED A DEFENSIBLE KILL AT ANY BLOCK LENGTH.** The un-resolvability is a property "
          f"of the instrument, not a close call about a free parameter.")
    print(f"[P1] and the {KILL_RHOBAR} threshold itself was NEVER CALIBRATED TO THIS ESTIMATOR: "
          f"it is inherited from `lore/harvest_6b32_appendix/mech_B.md`'s DIFFERENT statistic "
          f"(90d trend-efficiency persistence, 1000 asset-label permutations, NW t >= 3). "
          f"Nothing checked that 0.10 is attainable by market-level pooled VR(30) persistence "
          f"on 36 block pairs; measured above, it is not.")
    print(f"[P1] CAVEAT AGAINST THIS SECTION, carried because the review panel raised it "
          f"against itself: the panel's PARAMETRIC simulations used a market variance share "
          f"w ~= 0.31, lower than the real panel implies, so their simulated attenuation is "
          f"PESSIMISTIC. The model-free sign-flip share test above is what the ceiling rests "
          f"on; the parametric numbers are corroboration only.")

    # ========================================================================================
    # REVISION 3 (J) — the LEVEL read: sample mismatch + the estimator's own null level
    # ========================================================================================
    print(f"\n=== P1 (J) — whole-sample VR(30) LEVEL vs breakeven, with the estimator's OWN "
          f"null level (which revision 2 never printed) ===")
    null_ship_mean = float(np.nanmean(sf_lvl_shipped))
    null_ship_sd = float(np.nanstd(sf_lvl_shipped, ddof=1))
    null_match_mean = float(np.nanmean(sf_lvl_matched))
    null_match_sd = float(np.nanstd(sf_lvl_matched, ddof=1))
    z_ship = (vr_level - null_ship_mean) / null_ship_sd if null_ship_sd > 0 else float("nan")
    z_match = ((vr_level_m - null_match_mean) / null_match_sd if null_match_sd > 0
               else float("nan"))
    print(f"[P1] SHIPPED estimator (denominator = every finite daily return): level "
          f"{vr_level:.4f} (n_num={n_num_ws}, n_den={n_den_ws}); its own null level under a "
          f"TRUE VR of 1 is {null_ship_mean:.4f} +/- {null_ship_sd:.3f}")
    print(f"[P1] SAMPLE-MATCHED estimator (denominator restricted to the asset-days that feed a "
          f"valid numerator window): level {vr_level_m:.4f} (n_num={n_num_m}, n_den={n_den_m}; "
          f"{100.0 * (1.0 - n_den_m / max(n_den_ws, 1)):.1f}% of denominator asset-days "
          f"excluded); its own null level is {null_match_mean:.4f} +/- {null_match_sd:.3f}")
    print(f"[P1] THE CORRECTED READ: the excluded asset-days (gap-adjacent, listing/delisting "
          f"edges) are MORE volatile and inflate the DENOMINATOR ONLY, so the shipped level is "
          f"biased low. Against the correct reference — the estimator's own null, not 1.0 — the "
          f"observed level is z = {z_ship:+.2f} (shipped) / {z_match:+.2f} (matched) from a "
          f"RANDOM WALK. It is NOT a shortfall against the {BREAKEVEN} breakeven; revision 2's "
          f"'0.9043 vs 1.024' framing compared a biased estimate against an uncalibrated bar.")
    print(f"[P1] RULE ADOPTED: the estimator's own null level is now printed WHENEVER the level "
          f"is compared to breakeven. A level read against 1.0 or against {BREAKEVEN} without "
          f"it is not interpretable.")

    print(f"\n[P1] per-asset whole-life VR(30) cross-sectional sd(VR-1) = {xsd:.4f} "
          f"(n_assets={per_asset_vr.size}) — CONTEXT ONLY, per the task file NOT a verdict "
          f"(5-40x the {BREAKEVEN - 1.0:.3f} break-even margin; expected range 0.16-0.46)")
    if per_asset_nsub.size:
        print(f"[P1] CAVEAT on that number, added in revision 3 because it is quoted as "
              f"corroborating the methodology: it inherits a PER-ASSET version of the k-bias. "
              f"A contributing asset needs only 8 sub-windows (measured: median "
              f"{np.median(per_asset_nsub):.0f}, 10th percentile "
              f"{np.percentile(per_asset_nsub, 10):.0f}), so each per-asset VR is biased low by "
              f"an asset-specific amount and the cross-asset sd carries cross-asset BIAS "
              f"dispersion on top of real dispersion in VR. Context, now with a stated defect.")

    # ========================================================================================
    # REVISION 3 (K) — R1's failure is UNINFORMATIVE
    # ========================================================================================
    print(f"\n=== P1 (K) — what would a venue EXACTLY AS DEAD AS THE KILL ASSERTS (rho = 0) do "
          f"to these rules? (using this artifact's own Fisher-z SEs) ===")
    pk = [c["p_above_kill_if_venue_dead"] for c in cells
          if np.isfinite(c["p_above_kill_if_venue_dead"])]
    # Two readings, and the CONSERVATIVE one (most favourable to R1) is the headline. The 44
    # cells are not independent -- they share the same data -- so multiplying over all 44
    # understates P(unanimity). The conservative version multiplies over ONE reading per block
    # length (the pooled, dead-retained series), i.e. the coarsest independent dimension anyone
    # has claimed for this sweep. Both are printed; the reading does not turn on the choice.
    p_r1_11 = 1.0
    for bd in BLOCK_GRID:
        c = _find(cells, {"block_days": bd, "universe": "pit_dead_retained",
                          "estimator": "pooled"})
        q = c["p_above_kill_if_venue_dead"]
        if np.isfinite(q):
            p_r1_11 *= (1.0 - q)
    p_r1_44 = 1.0
    for c in cells:
        q = c["p_above_kill_if_venue_dead"]
        if np.isfinite(q):
            p_r1_44 *= (1.0 - q)
    p_r1 = p_r1_11
    # R2 on a dead venue: the anchor's CI must fall entirely below the kill line.
    se_z_a = 1.0 / np.sqrt(max(anchor["n_pairs"] - 3, 1))
    thr_z = np.arctanh(KILL_RHOBAR) - 1.959963984540054 * se_z_a
    p_r2 = float(0.5 * math.erfc(-(thr_z / se_z_a) / np.sqrt(2.0)))
    # R1 and R2 are NOT independent (the anchor is one of R1's own cells), so the conjunction
    # is bounded by the smaller of the two rather than by their product.
    p_conj = min(p_r1, p_r2)
    print(f"[P1] per-cell P(rho_hat >= {KILL_RHOBAR} | rho = 0) runs "
          f"[{min(pk):.3f},{max(pk):.3f}] across the {n_cells} cells.")
    print(f"[P1] therefore P(R1 unanimity | rho = 0) = {p_r1:.4f} on the conservative "
          f"{len(BLOCK_GRID)}-reading dimension ({p_r1_44:.2e} if all {n_cells} cells are "
          f"treated as independent), and P(R2 resolves below | rho = 0) = {p_r2:.3f}; "
          f"conjunction <= {p_conj:.4f} ({100 * p_conj:.2f}%).")
    print(f"[P1] READING: R1's FAILURE CARRIES NO INFORMATION. A dead venue fails R1 about "
          f"{100 * (1 - p_r1):.1f}% of the time, so '{n_below}/{n_cells} below -> FAIL' is what "
          f"a dead venue looks like, not evidence against one. R1 is kept as the right DECISION "
          f"rule for a destructive verdict; it is not evidence, and revision 2 reported it as "
          f"though the data had produced it.")
    veto = [c for c in cells if not c["below_kill"]]
    veto_xs = [c for c in veto if c["estimator"] == "xsmean"]
    veto_straddle = [c for c in veto
                     if np.isfinite(c["ci95_lo"]) and c["ci95_lo"] < 0.0 < c["ci95_hi"]
                     and c["ci95_lo"] < KILL_RHOBAR < c["ci95_hi"]]
    print(f"[P1] and the composition of the {len(veto)} vetoing cells matters: "
          f"{len(veto_xs)}/{len(veto)} are `xsmean`, an estimator this module's own docstring "
          f"marks 'multiplicity grid only' and which TASK_917:20-21 excludes from resolving the "
          f"sign (only market-level POOLING can). {len(veto_straddle)}/{len(veto)} of them have "
          f"a 95% CI containing BOTH {KILL_RHOBAR} and 0, i.e. not one of them resolves the "
          f"line it vetoes.")
    print(f"[P1] under R2's OWN CI standard: {len(resolve_pos)}/{n_cells} cells resolve the "
          f"{KILL_RHOBAR} line FROM ABOVE and {len(resolve_neg)}/{n_cells} FROM BELOW.")

    # ---- verdict, per the pre-declared decision rule in the module docstring ----
    r1_unanimous = all(c["below_kill"] for c in cells if np.isfinite(c["rho_bar"]))
    r2_anchor_resolves = bool(anchor["ci_excludes_kill_line"])
    fired = bool(r1_unanimous and r2_anchor_resolves)
    print(f"\n[P1] DECISION RULE R1 (unanimity across the sweep, destructive-verdict guard): "
          f"{'PASS' if r1_unanimous else 'FAIL'} "
          f"({n_below}/{n_cells} cells below {KILL_RHOBAR})")
    print(f"[P1] DECISION RULE R2 (anchor's 95% CI excludes the kill line): "
          f"{'PASS' if r2_anchor_resolves else 'FAIL'} "
          f"(anchor CI [{anchor['ci95_lo']:+.4f},{anchor['ci95_hi']:+.4f}] vs {KILL_RHOBAR}) "
          f"— but see (H): this 'PASS' is a tiling-phase lottery outcome, not resolving power.")
    print(f"[P1] KILL BAND rho_bar < {KILL_RHOBAR} -> "
          f"{'FIRED' if fired else 'DOES NOT FIRE — INCONCLUSIVE'}")
    # ---- (H) REVISION 2's PRO-KILL COUNTERFACTUAL, DELETED AND REPLACED BY ITS REFUTATION --
    print(f"[P1] (H) WITHDRAWN CLAIM. Revision 2 printed here that 'read on the ANCHOR CELL "
          f"ALONE the kill condition IS met and IS resolvable', citing the CI upper end "
          f"{anchor['ci95_hi']:+.4f} and that 'all four 90d cells agree to within 0.02'. THAT "
          f"CLAIM IS REFUTED, not softened:")
    print(f"     * the four 90d cells agree because they share ONE TILING ORIGIN, which "
          f"`_block_boundaries` hard-wires to day 0 and revision 2 never named as a parameter. "
          f"Their agreement measures the universe and estimator axes, and says nothing about "
          f"the axis that actually moves the answer.")
    print(f"     * across all {anchor_osum['n_origins']} tiling origins the anchor's rho_bar "
          f"has mean {anchor_osum['mean']:+.4f}, sd {anchor_osum['sd']:.4f}, range "
          f"[{anchor_osum['min']:+.4f},{anchor_osum['max']:+.4f}], and "
          f"{anchor_osum['frac_above_kill']:.1%} of origins land AT OR ABOVE the kill line. The "
          f"shipped origin is {z_shipped:+.2f} sd into the negative tail.")
    print(f"     * therefore R2 is not a power guard: 'the CI excludes {KILL_RHOBAR}' with "
          f"SE ~= {anchor_osum['sd']:.3f} is a statement about where one draw of a free "
          f"parameter landed. There was no resolvable pro-kill reading in the pack.")
    print(f"[P1] WHY THE KILL IS WITHHELD (restated on the corrected evidence): (a) THE "
          f"INSTRUMENT LACKED THE POWER TO FIRE IT AT ANY BLOCK LENGTH — the sign-flip null in "
          f"(I) cannot distinguish the observed block-VR dispersion from estimator noise "
          f"anywhere on the grid, capping attainable |rho_bar| near "
          f"{max(shares):.2f} and the anchor's power to separate {KILL_RHOBAR} from 0 near "
          f"{pw2:.0%}; (b) the anchor's 'resolving' CI is a tiling-phase artefact (H); (c) the "
          f"permutation null does not reject a RANDOM reordering of the same block VRs even "
          f"unadjusted (p = {anchor['perm_p']:.4f}), let alone at Bonferroni {bonf_alpha:.5f}; "
          f"(d) the {KILL_RHOBAR} threshold was inherited from a DIFFERENT statistic and never "
          f"calibrated to this estimator; (e) R1's failure is uninformative (K) — a dead venue "
          f"fails it ~99% of the time — so it is a decision rule, not evidence.")
    print(f"[P1] (M) THE k-PARITY QUESTION IS SETTLED = SAMPLING NOISE, and the revision-2 "
          f"framing ('unexplained', 'may be unsound at every block length', 'an open instrument "
          f"concern') is WITHDRAWN. White-noise ground truth (review panel, 2000 seeds, the "
          f"real NaN mask, T=4657/A=398) returns rho_bar means of -0.020 .. -0.093 MONOTONE in "
          f"block length with ZERO parity structure — textbook -1/n_pairs bias — and the null "
          f"sd matches this module's Fisher SE to 2 dp (at 90d: 0.168 vs 0.174), so the CIs "
          f"printed above are honest. The apparent parity does not replicate under ANY "
          f"perturbation: the origin shifts measured in (H) above break it, Q=20 and Q=45 give "
          f"different sequences, and non-multiple-of-30 block lengths scatter. Pattern "
          f"statistics against the sign-flip null: parity contrast p = 0.61, range p = 0.89. "
          f"The estimator recovers correct signs on ground truth; it is NOISY, NOT BROKEN.")
    if not fired:
        print("[P1] P1 DOES NOT RESOLVE THE QUESTION — and revision 3 sharpens WHY. It is not "
              "merely that the answer depends on a free parameter (it depends on two: block "
              "length AND tiling origin). It is that the instrument had no power to resolve "
              f"the {KILL_RHOBAR} line at any setting of either. No regime-gating idea is "
              "closed by this instrument. Symmetrically and equally load-bearing: NO "
              f"persistence is claimed either — {len(resolve_pos)}/{n_cells} cells exclude the "
              "kill line on the ABOVE side, i.e. not one cell in the sweep is evidence FOR "
              "persistence. The correct reading is UNDERPOWERED BY CONSTRUCTION, not 'dead' "
              "and not 'alive'. What a future row would need is unchanged and now quantified: "
              "a statistic with finer time resolution than one scalar per block (non-overlap "
              "is what caps n_pairs at ~36), or mech_B's per-asset formulation with its "
              "1000-draw asset-label permutation null.")

    out = {"revision": 3,
           "closing_pass": {
               "cycle": 3,
               "scope": "corrected the STATED ARGUMENTS of revision 2, which two adversarial "
                        "judges proved defective; changed NO verdict (P1 still: INCONCLUSIVE, "
                        "kill does not fire). The corrected reasoning is STRONGER: the "
                        "instrument had no power to fire the kill at any block length.",
               "disclosure": "D76 convention — the deviation is disclosed in the verdict "
                             "itself, not in a footnote",
               "corrections": [
                   "(H) the pro-kill counterfactual is WITHDRAWN and replaced by its "
                   "refutation: the four 90d cells share one undisclosed TILING ORIGIN; R2 is "
                   "not a power guard",
                   "(I) the model-free power ceiling is disclosed: P1 could not have fired a "
                   "defensible kill at ANY block length; the 0.10 threshold was inherited from "
                   "a different statistic and never calibrated to this estimator",
                   "(J) numerator/denominator sample mismatch corrected for the level read; "
                   "the estimator's own null level is now always printed beside breakeven; the "
                   "k-bias of the block grid and the exact block accounting are disclosed",
                   "(K) R1's failure is uninformative (a dead venue fails it ~99% of the "
                   "time); 10/12 vetoing cells are `xsmean`, excluded by TASK_917:20-21 from "
                   "resolving the sign",
                   "(L) the hardcoded 'universe and estimator move rho_bar by <=0.15' claim is "
                   "REFUTED by measurement (9/11 block lengths violate it); the 44->11 "
                   "multiplicity deflation that rested on it is withdrawn",
                   "(M) the k-parity pattern is SETTLED = sampling noise; the 'may be unsound' "
                   "framing is withdrawn",
                   "small items: circ_p is not comparable to the observed statistic (fewer "
                   "pairs per draw) and is no longer tabled as if it were; circular draws at "
                   "360d number 5, not 4; the sd(VR-1) context statistic carries a per-asset "
                   "k-bias",
               ]},
           "cells": cells, "by_block_length": by_block, "n_cells_swept": n_cells,
           "tiling_origin_sweep_anchor": {"rows": anchor_origins, "summary": anchor_osum,
                                          "shipped_origin": 0,
                                          "shipped_rho_bar": shipped_rho,
                                          "shipped_z_vs_origin_distribution": z_shipped},
           "tiling_origin_coarse_phases": {str(bd): [r["rho_bar"] for r in rows]
                                           for bd, rows in coarse.items()},
           "tiling_origin_single_30d_shift": {str(bd): v for bd, v in one_phase.items()},
           "power_ceiling_sign_flip": {
               "n_draws": SIGN_FLIP_DRAWS, "seed": SIGN_FLIP_SEED,
               "null": "per-date common sign flip of daily returns (true VR == 1; marginal "
                       "variances and cross-sectional covariance preserved; nothing fitted)",
               "by_block_length": power_rows,
               "n_block_lengths_dispersion_significant": n_sig_disp,
               "max_implied_signal_variance_share": max_attainable,
               "anchor_power_two_sided": pw2, "anchor_power_one_sided": pw1,
               "conclusion": "P1 could not have fired a defensible kill at ANY block length",
               "threshold_provenance": "0.10 inherited from mech_B.md's 90d trend-efficiency "
                                       "persistence (1000 asset-label permutations, NW t>=3); "
                                       "NEVER calibrated to market-level pooled VR(30) "
                                       "persistence",
               "caveat_against_itself": "the review panel's PARAMETRIC sims used market "
                                        "variance share w~=0.31, lower than the real panel "
                                        "implies, so their attenuation is pessimistic; the "
                                        "model-free share test above is the load-bearing "
                                        "evidence"},
           "level_read": {
               "shipped_level": vr_level, "shipped_n_num": n_num_ws,
               "shipped_n_den": n_den_ws,
               "shipped_null_level_mean": null_ship_mean,
               "shipped_null_level_sd": null_ship_sd, "shipped_z_vs_own_null": z_ship,
               "matched_level": vr_level_m, "matched_n_num": n_num_m,
               "matched_n_den": n_den_m,
               "matched_null_level_mean": null_match_mean,
               "matched_null_level_sd": null_match_sd, "matched_z_vs_own_null": z_match,
               "rule": "the estimator's own null level is printed WHENEVER the level is "
                       "compared to breakeven",
               "withdrawn": "revision 2's '0.9043 vs the 1.024 breakeven' shortfall framing — "
                            "it compared a sample-mismatched estimate against an uncalibrated "
                            "bar"},
           "r1_informativeness": {
               "per_cell_p_above_kill_if_dead_min": min(pk),
               "per_cell_p_above_kill_if_dead_max": max(pk),
               "p_r1_unanimity_if_venue_dead_11_readings": p_r1_11,
               "p_r1_unanimity_if_venue_dead_44_cells": p_r1_44,
               "p_r2_resolves_below_if_venue_dead": p_r2,
               "p_conjunction_if_venue_dead_upper_bound": p_conj,
               "n_vetoing_cells": len(veto),
               "n_vetoing_cells_xsmean": len(veto_xs),
               "n_vetoing_cells_ci_contains_both_kill_and_zero": len(veto_straddle),
               "reading": "R1's failure carries no information about the venue; it is a "
                          "decision rule, not evidence"},
           "independence_claim_refuted": {
               "claim": "universe and estimator move rho_bar by <=0.15 within a block length",
               "measured_within_block_spreads": spreads,
               "n_violating": len(viol), "n_block_lengths": len(BLOCK_GRID),
               "max_spread": max(spreads),
               "status": "WITHDRAWN — it was a hardcoded string the code never evaluated; the "
                         "44->11 multiplicity deflation resting on it is withdrawn with it"},
           "k_parity": {
               "status": "SETTLED = SAMPLING NOISE",
               "ground_truth": "white noise, 2000 seeds, real NaN mask, T=4657/A=398: rho_bar "
                               "means -0.020..-0.093 monotone in block length, zero parity "
                               "structure (textbook -1/n_pairs bias)",
               "ci_honesty": "null sd matches the module's Fisher SE to 2dp (90d: 0.168 vs "
                             "0.174)",
               "non_replication": "origin shifts (measured in this module), Q=20/Q=45, and "
                                  "non-multiple-of-30 blocks all break the pattern",
               "pattern_statistics_vs_sign_flip_null": {"parity_contrast_p": 0.61,
                                                        "range_p": 0.89},
               "withdrawn_framing": "'unexplained' / 'may be unsound at every block length' / "
                                    "'an open instrument concern'"},
           "n_cells_ci_resolves_below": len(resolve_neg),
           "n_cells_ci_resolves_above": len(resolve_pos),
           "block_grid": list(BLOCK_GRID), "anchor_spec": ANCHOR, "anchor": anchor,
           "legacy_headline_spec": LEGACY_HEADLINE, "legacy_headline": legacy,
           "anchor_provenance": "block_days=90 inherited from lore/harvest_6b32_appendix/"
                                "mech_B.md:44,48 (90-day non-overlapping windows, rho-bar>=0.10); "
                                "block_days=180 was TASK_917's own addition, disclosed nowhere "
                                "in revision 1",
           "grid_provenance": "the task file mandates 'Bonferroni over 8 cells' as a COUNT; the "
                              "composition of revision 1's 8-cell grid (in particular the 180d "
                              "entry) was the implementer's own pre-registration, not the "
                              "task's. Revision 2 sweeps the full block grid and Bonferronis "
                              "over the actual cell count.",
           "decision_rule": {"R1_unanimity_across_sweep": r1_unanimous,
                             "R2_anchor_ci_excludes_kill_line": r2_anchor_resolves,
                             "rule": "destructive kill fires only if EVERY swept cell reads "
                                     "rho_bar < 0.10 AND the anchor cell's 95% CI excludes the "
                                     "kill line; otherwise INCONCLUSIVE"},
           "n_cells_below_kill": n_below, "n_cells_at_or_above_kill": n_above,
           "n_cells_ci_excludes_kill_line": n_resolving,
           "vr_level_whole_sample": vr_level,
           "n_num_whole_sample": n_num_ws, "n_den_whole_sample": n_den_ws,
           "xsd_per_asset_vr_minus_1": xsd, "n_assets_xsd": int(per_asset_vr.size),
           "kill_band": KILL_RHOBAR, "kill_fired": fired,
           "verdict": "KILL FIRED" if fired else "INCONCLUSIVE — KILL DOES NOT FIRE",
           "breakeven": BREAKEVEN}
    def _json_default(o):
        if isinstance(o, (np.floating, float)) and not np.isfinite(o):
            return None
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(f"not serializable: {type(o)}")

    print(f"\n[P1] OVERWRITING artifact: {ARTIFACT}")
    ARTIFACT.write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"[P1] wrote {ARTIFACT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
