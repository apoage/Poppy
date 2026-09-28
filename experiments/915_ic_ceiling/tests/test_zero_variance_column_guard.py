#!/usr/bin/env python3
"""Regression guard for the real bug the 2026-09-28 timed dry run surfaced: F4's `f4_l1_3`
(the "time" path-dimension increment, a deterministic constant of the window length, not of
the asset) and every `f4_l2_i_i` (log-sig level-2 antisymmetric diagonal, always exactly 0)
have ZERO cross-sectional variance on every date. The old `cross_sectional_zscore` mapped a
zero-std date to NaN, which poisoned `fit_cell`'s joint-finiteness mask
(`np.isfinite(X).all(axis=-1)`) for EVERY row that included such a column -- F4 and ALL (64/160
cells) silently produced zero training/test rows on the real panel, indistinguishable from
"no signal" (NaN mean_ic) without this adversarial check. Proven live on gpu-server's real
15-panel dry run (`f4_l1_3` finite frac 0.0 in the z-scored panel pre-fix; joint finite frac
across all 84 F4 columns 0.0). This test proves the fix at the unit level: a zero-variance
column across live assets never NaNs out a JOINT finiteness check with a real, informative
column."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grid  # noqa: E402


def test_constant_column_does_not_poison_joint_finiteness():
    T, A = 50, 6
    present = np.ones((T, A), dtype=bool)
    rng = np.random.default_rng(0)
    real_signal = rng.normal(size=(T, A))
    constant_col = np.ones((T, A)) * 1.0          # identical value for every asset, every date
    z_real = grid.cross_sectional_zscore(real_signal, present)
    z_const = grid.cross_sectional_zscore(constant_col, present)
    assert np.all(z_const == 0.0)
    assert np.isfinite(z_real).all()
    stacked = np.stack([z_real, z_const], axis=-1)
    joint_finite = np.isfinite(stacked).all(axis=-1)
    assert joint_finite.all(), "a zero-variance column must not poison joint finiteness"


def test_zscore_still_nans_non_present_and_handles_mixed_dates():
    T, A = 4, 3
    present = np.array([[True, True, False],
                        [True, True, True],
                        [True, False, False],
                        [True, True, True]])
    x = np.array([[1.0, 2.0, 99.0],
                  [5.0, 5.0, 5.0],       # zero-variance date (all live assets equal)
                  [3.0, 99.0, 99.0],     # only one live asset -> std undefined (NaN)
                  [1.0, 4.0, 7.0]])
    z = grid.cross_sectional_zscore(x, present)
    assert np.isnan(z[0, 2])              # not present
    assert np.allclose(z[1, :], 0.0)      # zero-variance date -> 0 for all live assets
    assert z[2, 0] == 0.0                 # single live asset: nanstd([x])==0 -> zero-variance
    assert np.isnan(z[2, 1]) and np.isnan(z[2, 2])   # not present
    assert np.isfinite(z[3]).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} zero-variance guard tests passed")
