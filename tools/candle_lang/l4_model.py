"""
L4 panel member (TASK_789, branch C) — reconstruction-only conv-VQ tokenizer over the
6-channel vol-unit substrate.

This is the FSQ-VQ "language" L4: a 1D-conv encoder collapses one vol-unit window into a
small latent, FSQ quantizes it to a single integer token id (collapse-proof, no learned
codebook — fsq_core.FSQ), and a conv decoder reconstructs the window. The token id IS the
candle-language word for that window. Trained purely by masked reconstruction; the eval
battery (A0/A1/A2/A8/A9) judges it, NOT a predictive head (design §2, non-predictive).

WHY NO RevIN (vs the donor tools/tokenizer_c0/fsq.py OHLCTokenizer):
  The substrate (volunit.features_for_run) is ALREADY vol-normalized and rescale-invariant
  by construction (each channel is a causal log-ratio / causal EWMA sigma, or a wick fraction).
  A RevIN / per-window instance-norm would (a) reintroduce in-window look-ahead (the mean/std
  are computed over the whole window, so the norm of bar 0 depends on bar W-1), and (b) break
  the load-bearing rescale-invariance claim that the whole transfer-by-construction rests on:
  instance-norm centering is invariant to an additive shift, but the substrate's invariance is
  to a multiplicative price rescale c->lam*c, and re-centering per window would couple the token
  to absolute in-window level. So L4 feeds the raw 6-ch vol-unit window straight into the conv
  encoder with NO normalization. Decode reconstructs the same vol-unit features.

Architecture (contract):
  x (B,W,6)
    -> 1D-conv encoder over the TIME axis (nn.Conv, kernel ~3-5, conv_ch channels), causal-agnostic
       (reconstruction, not prediction) -> global-mean-pool over time -> Dense(hidden) -> Dense(d)
    -> fsq_core.FSQ(levels) -> zhat (B,d), idx (B,)
    -> Dense(hidden) -> Dense(W * conv_ch[-1]) -> reshape (B,W,conv_ch[-1]) -> conv decoder
       -> Dense/Conv head to (B,W,6) recon
  recon_loss(out, x, valid) = masked mean((recon-x)^2) over valid rows (tail-pad rows excluded).

CLAUDE.md JAX pitfalls honored: jnp.maximum division guards, no lax.cond in vmap, static shapes
(W is a module field -> reshape is static), STE lives in fsq_core.round_ste.
"""
import jax.numpy as jnp
import flax.linen as nn

from . import config as C          # package-relative (top-level module)
from .fsq_core import FSQ


class L4Tokenizer(nn.Module):
    """Reconstruction-only conv-VQ tokenizer over the 6-ch vol-unit substrate.

    Fields (contract):
        window      : int    window length W (static -> reshape is static-shape safe)
        n_feat      : int=6  vol-unit channel count (config.N_FEAT)
        hidden      : int=128 width of the bottleneck Dense layers
        fsq_levels  : tuple=(8,8,8,5,5)  implicit codebook = prod(levels) = 12800
        conv_ch     : tuple=(64,128)     1D-conv channel ladder (encoder; decoder mirrors)

    __call__(x (B,W,6)) -> dict(
        recon (B,W,6), z (B,d), zhat (B,d), idx (B,), codebook_size int)
      where d = len(fsq_levels).
    """
    window: int
    n_feat: int = C.N_FEAT
    hidden: int = 128
    fsq_levels: tuple = C.DEFAULT_FSQ_LEVELS
    conv_ch: tuple = (64, 128)

    @nn.compact
    def __call__(self, x):
        B = x.shape[0]
        W = self.window
        d = len(self.fsq_levels)
        dec_ch = self.conv_ch[-1]

        # ---- encoder: 1D conv ladder over the TIME axis (NHWC: [B, W, ch]) ----
        # NO RevIN / instance-norm — substrate is already vol-normalized & rescale-invariant.
        h = x
        for ch in self.conv_ch:
            h = nn.Conv(features=ch, kernel_size=(5,), padding="SAME")(h)
            h = nn.gelu(h)
        # global mean-pool over time -> window-summary vector (B, conv_ch[-1])
        pooled = jnp.mean(h, axis=1)
        g = nn.gelu(nn.Dense(self.hidden)(pooled))
        z = nn.Dense(d)(g)                                   # project to FSQ dims (B,d)

        # ---- FSQ quantize -> single integer token id ----
        fsq = FSQ(levels=tuple(self.fsq_levels))
        zhat, idx = fsq(z)                                   # zhat (B,d) in [-1,1], idx (B,)

        # ---- decoder: lift quant code -> per-time features -> conv reconstruction ----
        u = nn.gelu(nn.Dense(self.hidden)(zhat))
        u = nn.Dense(W * dec_ch)(u)
        u = u.reshape(B, W, dec_ch)                          # static W -> static shape
        for ch in reversed(self.conv_ch[:-1]):
            u = nn.Conv(features=ch, kernel_size=(5,), padding="SAME")(u)
            u = nn.gelu(u)
        recon = nn.Conv(features=self.n_feat, kernel_size=(5,), padding="SAME")(u)  # (B,W,6)

        return {
            "recon": recon,
            "z": z,
            "zhat": zhat,
            "idx": idx,
            "codebook_size": fsq.codebook_size,
        }


def recon_loss(out, x, valid):
    """Masked reconstruction MSE.

    Args:
        out   : dict from L4Tokenizer.__call__ (uses out["recon"], (B,W,6))
        x     : (B,W,6) target vol-unit features
        valid : (B,) bool — False for tail-pad rows (db_stream batch['valid']); excluded.

    Returns scalar mean of (recon-x)^2 over the VALID rows only (all W*6 elements per row).
    Division guarded per CLAUDE.md (jnp.maximum(.,1e-10)) so an all-pad batch can't NaN.
    """
    recon = out["recon"]
    se = (recon - x) ** 2                                    # (B,W,6)
    per_row = jnp.mean(se, axis=(1, 2))                      # (B,) mean over W*6 per window
    m = valid.astype(per_row.dtype)                          # (B,) 1.0 valid / 0.0 pad
    denom = jnp.maximum(jnp.sum(m), C.DIV_EPS)
    return jnp.sum(per_row * m) / denom
