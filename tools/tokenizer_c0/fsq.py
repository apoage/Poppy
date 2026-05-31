"""
Finite Scalar Quantization (FSQ, arXiv:2309.15505) in JAX/Flax, plus a small
RevIN-style instance-normalized patch tokenizer for OHLC shape windows.

FSQ summary: project encoder output to d dims, bound each dim to (-L/2, L/2)
via a scaled tanh, then round to the nearest integer with a straight-through
estimator. The implicit codebook size is prod(levels). No learned codebook ->
collapse-proof (the whole point of using FSQ for a robust tokenizer).

Design (locked):
  RevIN per-window instance-norm (mean/std over the window, per channel) ->
  flatten patch -> MLP encoder -> FSQ -> MLP decoder -> reconstruct normalized
  features. (mean,std) carried out-of-band and reapplied at decode time so we
  can recover de-normalized price paths for fidelity scoring.
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------- FSQ core ----------------

def round_ste(z):
    """Round with straight-through gradient."""
    return z + jax.lax.stop_gradient(jnp.round(z) - z)


class FSQ(nn.Module):
    levels: tuple  # e.g. (8,8,8,5,5,5) -> codebook = prod = 72000... pick to taste

    def setup(self):
        self._levels = jnp.asarray(self.levels, dtype=jnp.float32)
        # basis for mapping per-dim codes -> single integer token id
        basis = np.concatenate([[1], np.cumprod(np.asarray(self.levels[:-1]))])
        self._basis = jnp.asarray(basis, dtype=jnp.int32)
        self.codebook_size = int(np.prod(self.levels))

    def bound(self, z):
        """Bound each dim to roughly (-(L-1)/2, (L-1)/2) so that after rounding
        we hit the L integer levels symmetrically around 0."""
        half = (self._levels - 1) / 2.0
        # tanh keeps it in (-1,1); scale to half-width. offset handles even L.
        offset = jnp.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = jnp.tanh(z + offset) * half - offset * 0  # offset applied below
        return jnp.tanh(z) * half

    def quantize(self, z):
        """z: [..., d] continuous -> zhat in integer-centered grid (STE)."""
        zb = self.bound(z)
        zhat = round_ste(zb)
        return zhat

    def codes_to_indices(self, zhat):
        """Map quantized values to a single integer token id in [0, codebook)."""
        half = (self._levels - 1) / 2.0
        idx_per_dim = jnp.round(zhat + half).astype(jnp.int32)  # 0..L-1
        idx_per_dim = jnp.clip(idx_per_dim, 0, (self._levels.astype(jnp.int32) - 1))
        return jnp.sum(idx_per_dim * self._basis, axis=-1)

    def __call__(self, z):
        zhat = self.quantize(z)
        idx = self.codes_to_indices(zhat)
        return zhat, idx


# ---------------- Tokenizer (encoder + FSQ + decoder) ----------------

class OHLCTokenizer(nn.Module):
    window: int
    n_feat: int = 5
    hidden: int = 256
    fsq_levels: tuple = (8, 8, 8, 5, 5)  # prod = 12800 implicit codes

    @nn.compact
    def __call__(self, x):
        """x: [B, window, n_feat] raw scale-free features.
        Returns dict with reconstruction, token idx, quant vector, and the
        RevIN (mean,std) used (so caller can denorm).
        """
        B = x.shape[0]
        # RevIN-style per-window instance norm, per channel.
        mean = jnp.mean(x, axis=1, keepdims=True)            # [B,1,F]
        std = jnp.std(x, axis=1, keepdims=True) + 1e-5       # [B,1,F]
        xn = (x - mean) / std                                # normalized
        flat = xn.reshape(B, self.window * self.n_feat)

        d = len(self.fsq_levels)
        h = nn.gelu(nn.Dense(self.hidden)(flat))
        h = nn.gelu(nn.Dense(self.hidden)(h))
        z = nn.Dense(d)(h)                                   # project to FSQ dims

        fsq = FSQ(levels=self.fsq_levels)
        zhat, idx = fsq(z)

        g = nn.gelu(nn.Dense(self.hidden)(zhat))
        g = nn.gelu(nn.Dense(self.hidden)(g))
        recon_n = nn.Dense(self.window * self.n_feat)(g)
        recon_n = recon_n.reshape(B, self.window, self.n_feat)

        # de-normalize reconstruction back to feature space
        recon = recon_n * std + mean
        return {
            "recon_n": recon_n,
            "recon": recon,
            "xn": xn,
            "idx": idx,
            "zhat": zhat,
            "mean": mean,
            "std": std,
            "codebook_size": fsq.codebook_size,
        }
