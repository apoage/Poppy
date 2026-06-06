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
        fsq_levels  : tuple=(8,8,4)  implicit codebook = prod(levels) = 256 (TASK_796 grid-256)
        conv_ch     : tuple=(64,128)     1D-conv channel ladder (encoder; decoder mirrors)

    __call__(x (B,W,6)) -> dict(
        recon (B,W,6), z (B,d), z_bound (B,d), zhat (B,d), idx (B,), codebook_size int)
      where d = len(fsq_levels). z_bound is the post-bound PRE-round latent FSQ rounds (the target
      of the MAGVIT/LFQ usage-entropy loss); see fsq_core.level_centers for its bin geometry.

    pool (TASK_796): time-pool of the conv features to ONE token/window. "mean" (default) is the
      exact prior global-mean-pool (variance sink -> codebook collapse); "strided"/"attn" preserve
      more latent variance into z WITHOUT changing the one-token-per-window count (A8/A9 contract).
    """
    window: int
    n_feat: int = C.N_FEAT
    hidden: int = 128
    fsq_levels: tuple = C.DEFAULT_FSQ_LEVELS
    conv_ch: tuple = (64, 128)
    pool: str = "mean"            # {"mean","strided","attn"} time-pool to ONE token/window (TASK_796)
    pool_segments: int = 4        # strided pool: number of contiguous time segments to mean then combine

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

        # ---- time-pool the conv features to ONE window-summary vector (B, conv_ch[-1]) ----
        # The default global MEAN-pool over all W bars is a VARIANCE SINK: it shrinks the latent
        # variance ~1/sqrt(W), pushing z toward a degenerate center and starving the FSQ grid
        # (TASK_796 codebook collapse root cause). --pool {strided,attn} preserve more variance into
        # z WITHOUT changing the one-token-per-window count (A8/A9/fingerprint contract).
        pool = str(self.pool)
        if pool == "mean":
            # current default — exact prior behavior.
            pooled = jnp.mean(h, axis=1)
        elif pool == "strided":
            # mean over a few contiguous time SEGMENTS (each segment keeps more variance than the
            # full-window mean), then concat the segment summaries and learn a Dense combine back to
            # conv_ch[-1]. Segments chosen as evenly as possible; remainder bars fold into the last.
            n_seg = max(1, int(self.pool_segments))
            n_seg = min(n_seg, W)                            # static (W, n_seg are module fields)
            edges = [int(round(i * W / n_seg)) for i in range(n_seg + 1)]
            seg_means = []
            for s in range(n_seg):
                a, b = edges[s], max(edges[s + 1], edges[s] + 1)
                seg_means.append(jnp.mean(h[:, a:b, :], axis=1))   # (B, conv_ch[-1])
            cat = jnp.concatenate(seg_means, axis=-1)        # (B, n_seg*conv_ch[-1])
            pooled = nn.Dense(self.conv_ch[-1])(cat)         # combine -> (B, conv_ch[-1])
        elif pool == "attn":
            # single learned-query attention pool over the conv features (one query -> one token).
            # scores_t = <q, W_k h_t> / sqrt(ch); softmax over time; pooled = sum_t a_t (W_v h_t).
            ch = self.conv_ch[-1]
            q = self.param("attn_query", nn.initializers.normal(1.0), (ch,))
            k = nn.Dense(ch, name="attn_key")(h)             # (B, W, ch)
            val = nn.Dense(ch, name="attn_value")(h)         # (B, W, ch)
            scores = jnp.einsum("bwc,c->bw", k, q) / jnp.sqrt(jnp.maximum(float(ch), 1.0))  # (B, W)
            attn = nn.softmax(scores, axis=1)                # (B, W) over time
            pooled = jnp.einsum("bw,bwc->bc", attn, val)     # (B, ch)
        else:
            raise ValueError(f"unknown pool={pool!r} (expected mean|strided|attn)")

        g = nn.gelu(nn.Dense(self.hidden)(pooled))
        z = nn.Dense(d)(g)                                   # project to FSQ dims (B,d)

        # ---- FSQ quantize -> single integer token id ----
        fsq = FSQ(levels=tuple(self.fsq_levels))
        zhat, idx, z_bound = fsq(z)                          # zhat (B,d) in [-1,1], idx (B,), z_bound (B,d)

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
            "z": z,                 # raw pre-bound projection (B,d)
            "z_bound": z_bound,     # post tanh+shift bound, PRE-round — the value FSQ rounds (B,d)
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


def entropy_loss(z_bound, centers, mask, valid, temp=0.5):
    """MAGVIT-v2 / LFQ-style usage-balancing entropy loss on the FSQ pre-quant latent.

    Acts on z_bound (B,d) — the post tanh+shift bound, PRE-round continuous latent FSQ rounds — to
    PRESSURE the encoder to populate the FSQ grid (FSQ has no built-in usage pressure; recon-only +
    a high-capacity decoder lets the grid collapse, TASK_796 root cause). Mechanism (per FSQ dim i
    with L_i bin centers):

        q_i(z) = softmax_over_levels( -(z_bound_i - centers_i)^2 / temp )   # (B, d, max_L)

      a SOFT assignment of each sample to dim-i's L_i integer bin centers (level_centers; spacing
      1.0 in the bound space, so temp~0.5 makes q meaningful, not one-hot). Pad level slots (i>=L_i)
      are masked OUT of the softmax (set to -inf logit -> 0 prob).

        L_entropy = mean_{batch,dims} H( q_i | x )   -   mean_{dims} H( E_x[q_i] )

      Term 1 (per-sample assignment entropy, MINIMIZED): each window commits to a level (confident,
        low-entropy q) -> the encoder USES the grid resolution instead of hedging at the center.
      Term 2 (batch-marginal entropy, MAXIMIZED via the minus sign): the AVERAGE assignment must
        spread across ALL L_i levels -> forces the population to OCCUPY the whole codebook.
      Minimizing L_entropy = (confident per-sample) AND (uniform marginal) = the LFQ usage target.

    All entropies in NATS (natural log), guarded by jnp.maximum (CLAUDE.md JAX pitfall: log of 0).
    Only VALID rows (batch['valid']) enter both the per-sample mean and the batch marginal, so the
    tail-pad rows never bias usage. Both softmax branches are evaluated (no lax.cond in vmap); the
    -inf padding is applied via jnp.where, static shapes throughout (centers/mask are (d, max_L)).

    Args:
        z_bound : (B, d)        post-bound pre-round latent (out["z_bound"])
        centers : (d, max_L)    per-dim integer bin centers, pad with 0.0   (fsq_core.level_centers)
        mask    : (d, max_L)    1.0 for valid level slots, 0.0 for pad slots (fsq_core.level_centers)
        valid   : (B,)          bool — exclude tail-pad rows
        temp    : float         softmax temperature in integer-grid units (default 0.5)

    Returns scalar L_entropy (nats). Lower = more confident + more uniform usage.
    """
    z = z_bound[:, :, None]                                  # (B, d, 1)
    c = centers[None, :, :]                                  # (1, d, max_L)
    msk = mask[None, :, :]                                   # (1, d, max_L) broadcast over batch
    # soft-assignment logits: -(z - center)^2 / temp; mask pad level slots to -inf before softmax.
    t = jnp.maximum(float(temp), C.DIV_EPS)
    logits = -((z - c) ** 2) / t                            # (B, d, max_L)
    neg_inf = jnp.asarray(-1e30, dtype=logits.dtype)
    logits = jnp.where(msk > 0.0, logits, neg_inf)          # pad slots -> -inf (zero prob)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)  # stabilize
    e = jnp.exp(logits) * msk                                # (B, d, max_L); zero pad mass exactly
    q = e / jnp.maximum(jnp.sum(e, axis=-1, keepdims=True), C.DIV_EPS)   # (B, d, max_L) soft assign

    vm = valid.astype(q.dtype)                               # (B,)
    n_valid = jnp.maximum(jnp.sum(vm), C.DIV_EPS)

    # --- term 1: per-sample assignment entropy H(q_i | x), averaged over valid rows AND dims ---
    logq = jnp.log(jnp.maximum(q, C.DIV_EPS))               # guard log(0) (pad slots are 0 -> masked)
    h_per_sample = -jnp.sum(q * logq * msk, axis=-1)        # (B, d) entropy per (row, dim), nats
    h1 = jnp.sum(h_per_sample * vm[:, None]) / (n_valid * h_per_sample.shape[1])  # scalar mean

    # --- term 2: batch-marginal entropy H(E_x[q_i]) over VALID rows, averaged over dims ---
    q_bar = jnp.sum(q * vm[:, None, None], axis=0) / n_valid   # (d, max_L) marginal over valid rows
    logqbar = jnp.log(jnp.maximum(q_bar, C.DIV_EPS))
    h_marg = -jnp.sum(q_bar * logqbar * mask, axis=-1)        # (d,) marginal entropy per dim, nats
    h2 = jnp.mean(h_marg)                                     # scalar mean over dims

    return h1 - h2
