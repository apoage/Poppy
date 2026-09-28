#!/usr/bin/env python3
"""Unit tests for signature.py, per TASK_915's disclosure requirement (no iisignature/esig):
level-1 = total increment; level-2 antisymmetric part = Levy area on a hand-computed
piecewise-linear path; Chen/shuffle identity on concatenated paths; log-sig of a straight line
has zero terms above level 1. Run: trader_env/bin/python -m pytest experiments/915_ic_ceiling/tests
(server, wrapped) or `python experiments/915_ic_ceiling/tests/test_signature.py` for a plain
assert-based run with no pytest dependency."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import signature as sig  # noqa: E402


def test_level1_is_total_increment():
    path = np.array([[0.0, 0.0], [1.0, 0.3], [2.0, -0.7], [1.5, 1.1]])
    S = sig.path_signature(path)
    L1, _, _ = sig.log_signature(S)
    assert np.allclose(L1, path[-1] - path[0])


def test_level2_antisymmetric_is_hand_computed_levy_area():
    """L-shaped path (0,0)->(1,0)->(1,1): hand-derived via Chen's identity (not this module's
    code path in isolation -- see docstring derivation) gives S2=[[0.5,1],[0,0.5]],
    L2 = S2 - 0.5*S1(x)S1 = [[0,0.5],[-0.5,0]]. Independently: 0.5 is exactly the shoelace area
    of the triangle (0,0),(1,0),(1,1) closed back to (0,0) -- the signed area between the path
    and its chord."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    S_raw = sig.path_signature(path)
    L1, L2, L3 = sig.log_signature(S_raw)
    assert np.allclose(L2, [[0.0, 0.5], [-0.5, 0.0]]), L2
    # antisymmetry holds generically (algebraic identity, not a property of this path)
    assert np.allclose(L2, -L2.T)
    # shoelace cross-check of the enclosed area, computed independently of signature.py
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    area = 0.5 * sum(pts[k][0] * pts[k + 1][1] - pts[k + 1][0] * pts[k][1]
                      for k in range(len(pts) - 1))
    assert abs(area - 0.5) < 1e-12
    assert abs(L2[0, 1] - area) < 1e-12


def test_chen_identity_on_concatenated_subpaths():
    rng = np.random.default_rng(0)
    path = np.cumsum(rng.normal(size=(7, 3)), axis=0)
    path = np.vstack([np.zeros((1, 3)), path])
    full = sig.path_signature(path)
    k = 4
    part_a = sig.path_signature(path[: k + 1])          # points 0..k
    part_b = sig.path_signature(path[k:])               # points k..end
    concat = sig.chen_multiply(part_a, part_b)
    for a, b in zip(full, concat):
        assert np.allclose(a, b, atol=1e-10), (a, b)


def test_straight_line_logsig_zero_above_level1():
    """Multiple co-linear increments along a fixed direction v: S2 = 0.5*S1(x)S1 exactly
    (algebraic consequence of every dx_k being a scalar multiple of the same v), so the log-sig
    is zero at levels 2 and 3."""
    v = np.array([2.0, -1.0, 0.5])
    steps = np.array([0.0, 1.0, 1.6, 3.0, 3.2])          # uneven step sizes, same direction
    path = steps[:, None] * v[None, :]
    S = sig.path_signature(path)
    L1, L2, L3 = sig.log_signature(S)
    assert np.allclose(L1, path[-1] - path[0])
    assert np.allclose(L2, 0.0, atol=1e-10), L2
    assert np.allclose(L3, 0.0, atol=1e-10), L3


def test_group_inverse_is_identity():
    rng = np.random.default_rng(1)
    path = np.cumsum(rng.normal(size=(9, 2)), axis=0)
    S = sig.path_signature(path)
    I = sig.group_inverse(S)
    ident = sig.chen_multiply(S, I)
    for level, expect_zero in zip(ident, (True, True, True)):
        assert np.allclose(level, 0.0, atol=1e-9), level


def test_rolling_window_matches_direct_subpath_signature():
    rng = np.random.default_rng(2)
    path = np.cumsum(rng.normal(size=(20, 2)), axis=0)
    path = np.vstack([np.zeros((1, 2)), path])
    window = 6
    L1r, L2r, L3r = sig.rolling_log_signatures(path, window)
    for t in (window - 1, window + 3, path.shape[0] - 1):
        a = t - window + 1
        direct = sig.path_signature(path[a: t + 1])
        L1d, L2d, L3d = sig.log_signature(direct)
        assert np.allclose(L1r[t], L1d, atol=1e-9)
        assert np.allclose(L2r[t], L2d, atol=1e-9)
        assert np.allclose(L3r[t], L3d, atol=1e-8)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} signature tests passed")
