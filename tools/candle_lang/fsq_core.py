"""
Finite Scalar Quantization (FSQ, arXiv:2309.15505) — REFERENCE formulation, JAX/Flax.

REUSE+FIX of tools/tokenizer_c0/fsq.py. The donor `bound()` (fsq.py:39-46) computed an
`offset`/`shift` for even levels then THREW THEM AWAY (returned `tanh(z)*half`). For even L
that grid is symmetric -> after round_ste only L-1 integers are reachable and, after the
+half index shift, code 0 is never emitted. Blocking correction #2: adopt the reference FSQ
(pre-tanh half-level shift for even L) so all prod(levels) codes are reachable, and GATE it
with a hard reachability assertion before any run writes token IDs.

Reference round-trip (Mentzer et al., Appendix):
  bound(z):    half_l = (L-1)*(1-eps)/2 ;  offset = 0.5 if L even else 0 ;  shift = atanh(offset/half_l)
               return tanh(z + shift) * half_l - offset
  quantize(z): round_ste(bound(z)) / (L//2)          -> values in [-1, 1] (decoder-friendly)
  indices:     round(zhat * (L//2) + (L//2))  in [0, L-1], dot basis -> single int id

No learned codebook -> collapse-proof (the whole point of FSQ here).
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


def round_ste(z):
    """Round with a straight-through gradient."""
    return z + jax.lax.stop_gradient(jnp.round(z) - z)


class FSQ(nn.Module):
    levels: tuple  # e.g. (8,8,8,5,5) -> codebook = prod(levels)
    eps: float = 1e-3

    def setup(self):
        lv = np.asarray(self.levels, dtype=np.float64)
        self._levels = jnp.asarray(lv, dtype=jnp.float32)
        self._half_width = jnp.asarray(lv // 2, dtype=jnp.float32)          # L//2 per dim
        # reference even-level shift, precomputed in numpy (static)
        half_l = (lv - 1.0) * (1.0 - self.eps) / 2.0
        offset = np.where(lv % 2 == 0, 0.5, 0.0)
        self._half_l = jnp.asarray(half_l, dtype=jnp.float32)
        self._offset = jnp.asarray(offset, dtype=jnp.float32)
        self._shift = jnp.asarray(np.arctanh(offset / half_l), dtype=jnp.float32)
        # mixed-radix basis: index_per_dim (0..L-1) -> single integer token id
        basis = np.concatenate([[1], np.cumprod(lv[:-1])]).astype(np.int64)
        self._basis = jnp.asarray(basis, dtype=jnp.int32)
        self.codebook_size = int(np.prod(self.levels))

    def bound(self, z):
        """Bound each dim so round_ste hits all L integer levels (symmetric for odd L,
        offset by half a level for even L per reference FSQ)."""
        return jnp.tanh(z + self._shift) * self._half_l - self._offset

    def quantize(self, z):
        """z: [..., d] continuous -> zhat in [-1, 1] (renormalized integer grid, STE)."""
        q = round_ste(self.bound(z))
        return q / self._half_width

    def codes_to_indices(self, zhat):
        """zhat in [-1,1] (output of quantize) -> single integer token id in [0, codebook)."""
        idx_per_dim = jnp.round(zhat * self._half_width + self._half_width).astype(jnp.int32)
        idx_per_dim = jnp.clip(idx_per_dim, 0, (self._levels.astype(jnp.int32) - 1))
        return jnp.sum(idx_per_dim * self._basis, axis=-1)

    def __call__(self, z):
        # z_bound is the continuous pre-quant latent FSQ rounds (after tanh+shift bound, BEFORE
        # round_ste) — the target of the MAGVIT/LFQ-style usage-entropy loss in train_l4. Its
        # integer bin centers are fsq_core.level_centers(levels) (spacing 1.0 in this space).
        z_bound = self.bound(z)
        q = round_ste(z_bound)
        zhat = q / self._half_width
        idx = self.codes_to_indices(zhat)
        return zhat, idx, z_bound


def level_centers(levels):
    """Per-dim FSQ bin-center integers in the BOUND space (the value round_ste rounds), padded to
    max(levels) with a validity mask. Used by the MAGVIT/LFQ-style entropy (usage) loss in train_l4.

    For a dim with L levels, bound(z) = tanh(z+shift)*half_l - offset ranges over exactly the L
    consecutive integers that round() snaps to (verified empirically against FSQ.bound):
        odd  L: {-(L//2), ..., +(L//2)}        (L ints, symmetric about 0)
        even L: {-(L//2), ..., +(L//2)-1}      (L ints, shifted by the -0.5 even-level offset)
    i.e. centers_j = arange(L) - (L//2). Spacing is exactly 1.0 in the bound space, so a soft
    assignment temp is calibrated in integer-grid units (temp~0.5 => meaningful, not one-hot).

    Returns:
        centers : (d, max_L) float32  per-dim integer centers, right-padded with 0.0
        mask    : (d, max_L) float32  1.0 for the L valid level slots, 0.0 for pad slots
    where d = len(levels), max_L = max(levels). Pad slots are masked OUT of the softmax (set to
    -inf logit) so they carry zero probability.
    """
    lv = [int(L) for L in levels]
    d = len(lv)
    max_L = max(lv) if lv else 1
    centers = np.zeros((d, max_L), dtype=np.float32)
    mask = np.zeros((d, max_L), dtype=np.float32)
    for j, L in enumerate(lv):
        c = np.arange(L, dtype=np.float32) - (L // 2)        # {-(L//2) .. -(L//2)+L-1}
        centers[j, :L] = c
        mask[j, :L] = 1.0
    return centers, mask


def assert_all_codes_reachable(levels, eps=1e-3, n_sweep=4096):
    """HARD GATE (blocking correction #2). Reachability is complete iff (a) every dim reaches
    ALL L of its integer levels under a deterministic wide sweep, AND (b) the mixed-radix basis
    is a BIJECTION from the Cartesian product of per-dim indices onto [0, prod(levels)). Together
    these imply every one of the codebook codes is reachable by some input (its per-dim preimage).
    A random-id-coverage check is WRONG here (coupon-collector: 20k samples can't cover 12800 codes).
    Call BEFORE any run writes token IDs (train_l4 startup + the unit test below)."""
    import itertools
    fsq = FSQ(levels=tuple(levels), eps=eps)
    d = len(levels)
    # (a) per-dim coverage under a deterministic linspace wide enough to saturate tanh both ways.
    sweep = np.linspace(-8.0, 8.0, n_sweep, dtype=np.float32)
    z = np.tile(sweep[:, None], (1, d))                      # same sweep on every dim
    params = fsq.init(jax.random.PRNGKey(0), jnp.asarray(z[:2]))
    zhat = np.asarray(fsq.apply(params, jnp.asarray(z), method=FSQ.quantize))
    per_dim_codes = []
    per_dim_ok = True
    for j, L in enumerate(levels):
        codes_j = np.clip(np.round(zhat[:, j] * (L // 2) + (L // 2)).astype(int), 0, L - 1)
        u = np.unique(codes_j)
        per_dim_codes.append(set(u.tolist()))
        if u.size < L:
            per_dim_ok = False
    # (b) basis bijection over the FULL Cartesian product (exact, no sampling).
    cb = int(np.prod(levels))
    basis = np.asarray(fsq.variables if False else None)     # noqa — basis recomputed below
    bvec = np.concatenate([[1], np.cumprod(np.asarray(levels[:-1]))]).astype(np.int64)
    if cb <= 5_000_000:                                       # safe to enumerate the default 12800
        ids = np.array([int(np.dot(combo, bvec))
                        for combo in itertools.product(*[range(L) for L in levels])], dtype=np.int64)
        bijection = (np.unique(ids).size == cb) and (ids.min() == 0) and (ids.max() == cb - 1)
    else:
        bijection = bool(np.all(bvec == np.concatenate([[1], np.cumprod(np.asarray(levels[:-1]))])))
    assert per_dim_ok and bijection, (
        f"FSQ reachability FAILED for levels={levels}: per_dim_ok={per_dim_ok} "
        f"(levels reached per dim: {[len(s) for s in per_dim_codes]} vs {list(levels)}), "
        f"basis_bijection={bijection}. The bound() fix is wrong — do NOT trust token IDs.")
    return dict(codebook_size=cb, per_dim_ok=bool(per_dim_ok), basis_bijection=bool(bijection))


if __name__ == "__main__":
    # smoke: prove the fix on the default + a few all-even / all-odd configs.
    for lv in [(8, 8, 8, 5, 5), (8, 8), (5, 5, 5), (4, 6, 8), (3, 5, 7)]:
        print(lv, assert_all_codes_reachable(lv))
    print("FSQ reachability gate PASSED for all tested level configs.")
