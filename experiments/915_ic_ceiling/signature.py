#!/usr/bin/env python3
"""TASK_915 F4 — truncated (level <=3) log-signature of a d-dim path, via Chen's identity.

No `iisignature`/`esig` on gpu-server's `trader_env` (verified 2026-09-28) -- implemented
directly in numpy. Disclosed substitution (task instruction): the truncated tensor algebra up
to level 3, Chen's identity for concatenation, and the standard log map for a truncated
group-like tensor series.

Representation: a truncated signature/log-signature of a d-dim path is a tuple
(S1, S2, S3) with S1 in R^d, S2 in R^(d,d), S3 in R^(d,d,d) (S0 == 1 is implicit and dropped).

Chen's identity (concatenation = truncated tensor product in the tensor algebra):
  (A*B)_1 = A1 + B1
  (A*B)_2 = A2 + A1 (x) B1 + B2
  (A*B)_3 = A3 + A2 (x) B1 + A1 (x) B2 + B3

One linear segment with increment dx (the signature of exp(dx), i.e. of a straight line):
  S1 = dx ; S2 = dx(x)dx/2 ; S3 = dx(x)dx(x)dx/6
so S(path) = chen-product of every segment's one-step signature, in order.

Group inverse (antipode), solved order-by-order so that S * inv(S) = 1:
  I1 = -S1
  I2 = S1(x)S1 - S2
  I3 = -S3 + S2(x)S1 + S1(x)S2 - S1(x)S1(x)S1

Rolling-window signature via the prefix trick (Chen's identity again):
  sig(path[a:b]) = inv(sig(path[:a])) * sig(path[:b])
so a length-T path needs one O(T) forward pass (cumulative prefix signature) plus one
inverse+multiply per window endpoint, rather than O(T*window) work.

Log map (truncated, standard formula for a group-like element S = 1 + T, T = S - 1):
  log(S) = T - T(x)T/2 + T(x)T(x)T/3   (truncated to level <=3; T has no level-0 part)
  log1 = S1
  log2 = S2 - (1/2) S1(x)S1                      -- EXACTLY antisymmetric (Levy area x2)
  log3 = S3 - (1/2)(S1(x)S2 + S2(x)S1) + (1/3) S1(x)S1(x)S1

Batched over an (N, d) array of increments / an (N, A, d) array of assets -- all tensor
products below are done with `einsum` and are vectorised over any leading batch shape.
"""
import numpy as np


def _outer2(a, b):
    """batched a (x) b, a,b: (..., d) -> (..., d, d)."""
    return a[..., :, None] * b[..., None, :]


def _outer3(a, b, c):
    """batched a (x) b (x) c -> (..., d, d, d)."""
    return a[..., :, None, None] * b[..., None, :, None] * c[..., None, None, :]


def _contract21(A2, B1):
    """A2 (x) B1 for A2: (...,d,d), B1: (...,d) -> (...,d,d,d) [(A2(x)B1)_ijk = A2_ij B1_k]."""
    return A2[..., :, :, None] * B1[..., None, None, :]


def _contract12(A1, B2):
    """A1 (x) B2 -> (...,d,d,d): (A1(x)B2)_ijk = A1_i B2_jk."""
    return A1[..., :, None, None] * B2[..., None, :, :]


def one_step_signature(dx):
    """dx: (..., d). Returns (S1, S2, S3) = signature of exp(dx) truncated at level 3."""
    S1 = dx
    S2 = 0.5 * _outer2(dx, dx)
    S3 = (1.0 / 6.0) * _outer3(dx, dx, dx)
    return S1, S2, S3


def chen_multiply(A, B):
    """A, B: (S1,S2,S3) tuples of matching batch shape. Returns A*B (truncated level<=3)."""
    A1, A2, A3 = A
    B1, B2, B3 = B
    C1 = A1 + B1
    C2 = A2 + _outer2(A1, B1) + B2
    C3 = A3 + _contract21(A2, B1) + _contract12(A1, B2) + B3
    return C1, C2, C3


def group_inverse(S):
    """Antipode: I such that S*I == identity (1,0,0)."""
    S1, S2, S3 = S
    I1 = -S1
    I2 = _outer2(S1, S1) - S2
    I3 = -S3 + _contract21(S2, S1) + _contract12(S1, S2) - _outer3(S1, S1, S1)
    return I1, I2, I3


def log_signature(S):
    """Truncated log map of a group-like S=(S1,S2,S3). Level-2 output is exactly
    antisymmetric (== the doubled Levy-area matrix); this is checked by test_signature.py."""
    S1, S2, S3 = S
    L1 = S1
    L2 = S2 - 0.5 * _outer2(S1, S1)
    L3 = S3 - 0.5 * (_contract21(S2, S1) + _contract12(S1, S2)) + (1.0 / 3.0) * _outer3(S1, S1, S1)
    return L1, L2, L3


def path_signature(path):
    """path: (T, d) (or (T, A, d) batched over assets). Returns the FULL-path signature
    (S1,S2,S3) via a python loop over T (once, O(T)); increments are path[1:]-path[:-1]."""
    dx = np.diff(path, axis=0)
    S = one_step_signature(dx[0])
    for i in range(1, dx.shape[0]):
        S = chen_multiply(S, one_step_signature(dx[i]))
    return S


def prefix_signatures(path):
    """path: (T, d) or (T, A, d). Returns list of length T of (S1,S2,S3), prefix_sig[t] =
    signature of path[0:t+1] (i.e. of the t increments path[1]-path[0] .. path[t]-path[t-1]);
    prefix_sig[0] is the identity (0 increments so far)."""
    T = path.shape[0]
    d_shape = path.shape[1:]
    ident = (np.zeros(d_shape), np.zeros(d_shape + d_shape), np.zeros(d_shape + d_shape + d_shape))
    out = [ident]
    if T < 2:
        return out
    dx = np.diff(path, axis=0)
    cur = one_step_signature(dx[0])
    out.append(cur)
    for i in range(1, dx.shape[0]):
        cur = chen_multiply(cur, one_step_signature(dx[i]))
        out.append(cur)
    return out


def rolling_log_signatures(path, window):
    """path: (T, d). Returns log-sig arrays (T,d), (T,d,d), (T,d,d,d) for the window ending at
    each t (inclusive), NaN-filled for t < window-1 (not enough history). Uses the prefix trick:
    sig(path[t-window+1 : t+1]) = inv(prefix[t-window]) * prefix[t]."""
    T, d = path.shape
    prefix = prefix_signatures(path)
    L1 = np.full((T, d), np.nan)
    L2 = np.full((T, d, d), np.nan)
    L3 = np.full((T, d, d, d), np.nan)
    for t in range(window - 1, T):
        a = t - window + 1
        # prefix[k] covers points 0..k (k increments). The window [a..t] (window points,
        # window-1 increments) has signature inv(prefix[a]) * prefix[t].
        seg = chen_multiply(group_inverse(prefix[a]), prefix[t])
        s1, s2, s3 = log_signature(seg)
        L1[t], L2[t], L3[t] = s1, s2, s3
    return L1, L2, L3
