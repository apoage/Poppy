"""
train_l4.py — leak-proof single-GPU training harness for the L4 FSQ-VQ panel member
(TASK_789, branch C).

L4 is the reconstruction-only conv-VQ "language": a 1D-conv encoder collapses one vol-unit
window into a small latent, FSQ quantizes it to a single integer token id (collapse-proof, no
learned codebook), and a conv decoder reconstructs the window. We train purely by masked
reconstruction over the streamed vol-unit substrate, then FREEZE and judge the tokenizer with
the shared non-predictive battery (eval_battery A0/A1/A2/A9 + A8 if holdout assets present) —
NOT a predictive head (the donor's RQ0 sign-probe is deleted, design §2 non-predictive).

LEAK / OOM CONTRACT (this file's whole reason to exist; build plan 04 + leak/OOM audit 05):
  (1) fsq_core.assert_all_codes_reachable(levels) is a HARD GATE at startup, BEFORE any training
      (so we never mint token IDs on a broken FSQ grid — blocking correction #2).
  (2) We DO NOT set XLA_PYTHON_CLIENT_PREALLOCATE anywhere. safe_run owns MEM_FRACTION (via its
      --gpu-frac -> XLA_PYTHON_CLIENT_MEM_FRACTION); PREALLOCATE stays default-ON. Removing the
      donor run_gate.py:119 `PREALLOCATE=false` line is the blocking fix (preallocation is the
      correct, lower-fragmentation path here per feedback_jax_vram_preallocate). We only
      setdefault MEM_FRACTION from --gpu-frac so a bare invocation still has a cap; if safe_run
      already exported it, setdefault leaves it untouched.
  (3) The jit train step donates the BATCH BUFFER xb ONLY (donate_argnums on xb), NEVER params or
      opt_state — donating those would free live state and corrupt training.
  (4) jax.block_until_ready after each step (train_phase2.py:2142) so the host can't run ahead of
      the device and pile up un-freed async buffers.
  (5) We consume db_stream.BatchStreamer (the bounded queue.Queue IS the RAM cap) and mask the
      loss by batch['valid'] so the tail-pad rows (added so the jit step never retraces) are
      excluded from the gradient.
  (6) n_dev = shard.n_devices(); single-GPU is the default. Multi-device is the batch axis only.
  (7) After training: freeze, encode a HELD-OUT stream with an ONLINE np.bincount (chunked
      np.asarray per encode call, run_gate.py:156-189 idiom) — NEVER a full host token list — so
      the encode pass is host-memory O(codebook), not O(windows) (eval_battery blocking #3).

CLAUDE.md JAX pitfalls honored: jnp.maximum guards live in l4_model.recon_loss; static shapes
(the streamer pads to cfg.batch so the jit step never retraces); no lax.cond in vmap; block_until_ready
before device->host transfers.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

# Package-relative imports (top-level modules + the symbolic-free siblings).
from . import config as C
from . import shard
from . import eval_battery as EB
from .db_stream import BatchStreamer, survey_row_counts
from .fsq_core import assert_all_codes_reachable
from .l4_model import L4Tokenizer, recon_loss
from .result_writer import write_result
from .volunit import features_for_run


# TASK_796 codebook-health floor: A9 FAILS if fewer than this fraction of codes are ever used over
# the held-out encode. The v1 grid-12800 collapse used 172/12800 = 1.3%; with the grid-256 default
# (8,8,4) and healthy training, usage should reach ~70-100%. A 0.10 floor (>= ~26/256 codes) is
# deliberately conservative — it catches a true collapse without flagging an asset set that simply
# doesn't exercise the long tail. Tighten once grid-256 usage is empirically characterized.
USED_FRAC_FLOOR = 0.10


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the L4 FSQ-VQ candle-language tokenizer (reconstruction-only)."
    )
    p.add_argument("--db", type=str, default=None,
                   help="SQLite DB path (default: config.db_path() / $CANDLE_DB).")
    p.add_argument("--window", type=int, default=C.DEFAULT_WINDOW,
                   help="vol-unit window length W.")
    p.add_argument("--fsq-levels", type=str, default=",".join(str(x) for x in C.DEFAULT_FSQ_LEVELS),
                   help="comma-separated FSQ levels, e.g. 8,8,8,5,5.")
    p.add_argument("--epochs", type=int, default=8,
                   help="number of passes over the build stream.")
    p.add_argument("--batch", type=int, default=C.DEFAULT_BATCH,
                   help="batch size (== cfg.batch; the streamer pads the tail to this).")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--max-windows-per-asset", type=int, default=40000,
                   help="cap windows kept per asset (most-recent kept); None-like 0 = unbounded.")
    p.add_argument("--gpu-frac", type=float, default=0.5,
                   help="per-device VRAM fraction; only setdefault'd into "
                        "XLA_PYTHON_CLIENT_MEM_FRACTION (safe_run wins if it already set it). "
                        "We NEVER set XLA_PYTHON_CLIENT_PREALLOCATE.")
    p.add_argument("--hidden", type=int, default=128, help="bottleneck Dense width.")
    p.add_argument("--conv-ch", type=str, default="64,128", help="comma-separated conv channel ladder.")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed for param init.")
    p.add_argument("--encode-chunk", type=int, default=4096,
                   help="host chunk size for the online-bincount encode pass.")
    p.add_argument("--out", type=str,
                   default="/home/lukas/Trader/runs/789_token_c0/L4_RESULT.md",
                   help="RESULT.md output path (a .json sidecar is written alongside).")
    # TASK_796 scale-up: explicit asset selection + per-asset transfer matrix (for the 150/150 run).
    p.add_argument("--build-assets", nargs="+", type=int, default=None,
                   help="explicit BUILD asset ids; default None => config.DEFAULT_BUILD_ASSETS.")
    p.add_argument("--holdout-assets", nargs="+", type=int, default=None,
                   help="explicit HOLDOUT asset ids; default None => config.DEFAULT_HOLDOUT_ASSETS.")
    p.add_argument("--transfer-matrix", action="store_true",
                   help="compute the per-asset A-by-B cross-asset transfer matrix in A8.")
    return p.parse_args(argv)


def _parse_int_tuple(s: str) -> tuple:
    return tuple(int(x) for x in str(s).split(",") if str(x).strip() != "")


# --------------------------------------------------------------------------- #
# A2 reconstruction-pair builder (price path from recon log-returns; run_gate.py:194-204 idiom)
# --------------------------------------------------------------------------- #
def _recon_pairs_for_batch(recon_np: np.ndarray, C_np: np.ndarray, valid_np: np.ndarray):
    """Yield (recon_price_path, true_price_path) pairs for the VALID rows of one batch.

    The vol-unit channel 0 (mv) is the volatility-normalized inter-bar log return. We rebuild a
    denormalized price path the same way the donor does (run_gate.py:194-204): start at the first
    true close of the window, then multiply by exp(cumsum(recon_mv * sigma))... but we do NOT have
    per-bar sigma here on the host. Instead we compare RELATIVE path shape directly: A2 is itself
    rescale-invariant (it folds log(recon/true)), so we feed the cumulative-product path of the
    reconstructed RELATIVE moves vs the true close path. We approximate the per-bar move scale by
    the realized window step ratio so the comparison is on a multiplicative (scale-free) footing.

    recon_np : (B, W, 6) float32 reconstructed vol-unit features
    C_np     : (B, W)    float32 raw true close path over the window (db_stream batch['C'])
    valid_np : (B,)      bool

    Yields one (recon_path(W-1,), true_path(W-1,)) pair per valid row. Streamed — never collected.
    """
    B, W, _ = recon_np.shape
    for b in range(B):
        if not bool(valid_np[b]):
            continue
        true_close = C_np[b].astype(np.float64)            # (W,)
        if W < 2:
            continue
        start = max(float(true_close[0]), C.EPS)
        # true per-bar realized log-return (defines the local vol scale we reproject onto).
        true_lr = np.log(np.maximum(true_close[1:], C.EPS) / np.maximum(true_close[:-1], C.EPS))  # (W-1,)
        true_scale = np.sqrt(np.mean(true_lr * true_lr)) if true_lr.size else 0.0
        true_scale = max(float(true_scale), C.DIV_EPS)
        # reconstructed vol-unit move (channel 0) -> reproject to a log-return via the window scale.
        recon_mv = recon_np[b, 1:, 0].astype(np.float64)   # (W-1,) recon mv for bars 1..W-1
        recon_lr = recon_mv * true_scale
        recon_path = start * np.exp(np.cumsum(recon_lr))    # (W-1,)
        true_path = true_close[1:]                          # (W-1,)
        yield recon_path, true_path


# --------------------------------------------------------------------------- #
# A0 tokenize-one-window closure (raw OHLC -> token id, full member path)
# --------------------------------------------------------------------------- #
def _make_tokenize_window_fn(apply_encode, params, cfg):
    """Return a callable (ohlc(W,4) raw price float) -> token_id (int) implementing L4's FULL
    tokenize path for ONE window: build vol-unit features (numpy canonical substrate) -> conv-VQ
    encode -> FSQ idx. Used by eval_battery.a0_rescale_invariance, which feeds rescaled copies and
    requires byte-identical token ids.

    The substrate is rescale-invariant by construction; A0 verifies the END-TO-END path (substrate
    + model) preserves that, which is the load-bearing claim.
    """
    import jax.numpy as jnp

    W = int(cfg.window)

    def tokenize(ohlc):
        ohlc = np.asarray(ohlc, dtype=np.float64)          # (Wp1, 4) o,h,l,c  (Wp1 >= W+1)
        Wp1 = ohlc.shape[0]
        # synthetic contiguous dt==60 timestamps for ONE gap-free run (A0 sample is gap-free).
        ts = (np.arange(Wp1, dtype=np.int64) * C.BAR_SECONDS) + 1_600_000_000
        feats, _close, _ts_out, finite = features_for_run(
            ts, ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], cfg
        )                                                  # feats (Wp1-1, 6)
        # take the most-recent W finite feature bars (one window).
        if feats.shape[0] < W:
            # pad-front with zeros (degenerate guard; A0 sample windows should be long enough).
            pad = np.zeros((W - feats.shape[0], C.N_FEAT), dtype=np.float32)
            win = np.concatenate([pad, feats], axis=0)[-W:]
        else:
            win = feats[-W:]
        win = np.ascontiguousarray(win[None, :, :], dtype=np.float32)   # (1, W, 6)
        idx = apply_encode(params, jnp.asarray(win))                   # (1,)
        return int(np.asarray(idx)[0])

    return tokenize


def _sample_ohlc_windows(cfg, assets, n=16):
    """Pull a SMALL diagnostic sample of raw-price OHLC windows (each (W+1, 4)) for A0.

    Uses db_stream.iter_runs to grab one gap-free run per asset and slices the most-recent W+1 raw
    bars. A0 is a unit check (not a stream fold), so this is intentionally tiny and bounded.
    """
    from .db_stream import iter_runs

    W = int(cfg.window)
    out = []
    for a in assets:
        try:
            for ts, o, h, l, c in iter_runs(cfg.db, int(a), cfg):
                if len(ts) >= W + 1:
                    ohlc = np.stack([o[-(W + 1):], h[-(W + 1):], l[-(W + 1):], c[-(W + 1):]], axis=1)
                    out.append(np.ascontiguousarray(ohlc, dtype=np.float64))
                    break
        except Exception:
            continue
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# online-bincount encode pass (host memory O(codebook), NEVER O(windows))
# --------------------------------------------------------------------------- #
def _encode_stream_online(apply_encode, params, cfg, assets, codebook_size,
                          max_windows_per_asset, chunk):
    """Encode a held-out stream and fold token ids into an O(codebook) bincount accumulator,
    streaming A2 reconstruction pairs and A9 length counts alongside.

    Returns (counts(codebook,), token_id_chunks_list_GENERATOR_unsupported...) — instead we return
    pre-folded artifacts so the caller never holds a full token list:
        counts      : (codebook_size,) int64  online token histogram
        n_tokens    : int                      total valid tokens encoded
        a2_pairs    : list of (recon_path, true_path) — BOUNDED (capped) sample for A2
        len_counts  : np.bincount-style length histogram (fixed-length member -> single spike at W)

    Token ids are chunked per-batch via np.asarray (run_gate.py:156-189 idiom): we device->host
    transfer ONE batch of ids at a time, bincount it into the accumulator, and DROP it. No host
    list of the whole stream is ever built.
    """
    import jax

    counts = np.zeros(int(codebook_size), dtype=np.int64)
    n_tokens = 0
    # A2 is a streamed mean in eval_battery, but we cap the materialized pair sample to keep the
    # encode pass bounded; a few thousand pairs is plenty for a faithful mean.
    a2_cap = 4096
    a2_pairs = []
    W = int(cfg.window)

    mwpa = None if (max_windows_per_asset is None or int(max_windows_per_asset) <= 0) \
        else int(max_windows_per_asset)
    streamer = BatchStreamer(cfg, assets, shuffle=False, max_windows_per_asset=mwpa)
    try:
        for batch in streamer:
            xb = batch["X"]                                # (B,W,6) np.float32
            valid = batch["valid"]                         # (B,) np.bool_
            out = apply_encode(params, jax.numpy.asarray(xb), want_recon=len(a2_pairs) < a2_cap)
            # out is (idx,) or (idx, recon) depending on want_recon
            if isinstance(out, tuple):
                idx_dev, recon_dev = out
                recon_np = np.asarray(recon_dev)
            else:
                idx_dev = out
                recon_np = None
            ids = np.asarray(idx_dev)                      # (B,) device->host, ONE batch only
            v = np.asarray(valid)
            valid_ids = ids[v]
            if valid_ids.size:
                # online accumulate (clip-guarded inside eval_battery; here a plain bincount).
                lo = int(valid_ids.min()); hi = int(valid_ids.max())
                if lo < 0 or hi >= int(codebook_size):
                    valid_ids = np.clip(valid_ids, 0, int(codebook_size) - 1)
                counts += np.bincount(valid_ids.astype(np.int64),
                                      minlength=int(codebook_size)).astype(np.int64)
                n_tokens += int(valid_ids.size)
            if recon_np is not None and len(a2_pairs) < a2_cap:
                for pair in _recon_pairs_for_batch(recon_np, np.asarray(batch["C"]), v):
                    a2_pairs.append(pair)
                    if len(a2_pairs) >= a2_cap:
                        break
    finally:
        streamer.close()

    # fixed-length member: every emitted token covers exactly W feature bars -> single length spike.
    len_counts = np.zeros(W + 1, dtype=np.int64)
    len_counts[W] = int(n_tokens)
    return counts, n_tokens, a2_pairs, len_counts


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)

    # ---- (2) VRAM env: setdefault MEM_FRACTION only; NEVER touch PREALLOCATE ----
    # safe_run already exports XLA_PYTHON_CLIENT_MEM_FRACTION via --gpu-frac; setdefault means we
    # don't clobber it. We deliberately do NOT set XLA_PYTHON_CLIENT_PREALLOCATE (stays default-on).
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.gpu_frac))

    levels = _parse_int_tuple(args.fsq_levels)
    conv_ch = _parse_int_tuple(args.conv_ch)
    max_windows = None if int(args.max_windows_per_asset) <= 0 else int(args.max_windows_per_asset)

    # ---- (1) HARD GATE: FSQ reachability BEFORE anything touches the device or mints ids ----
    gate = assert_all_codes_reachable(levels)
    print(f"[gate] FSQ reachability PASSED: {gate}", flush=True)

    # JAX imports AFTER the env is set (so MEM_FRACTION applies to the first device init).
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state

    print("JAX devices:", jax.devices(), flush=True)
    n_dev = shard.n_devices()                              # (6) single-GPU default
    print(f"[shard] n_devices={n_dev}", flush=True)

    db = args.db if args.db else C.db_path()
    # TASK_796: explicit asset selection overrides the config defaults; a bare invocation (both None)
    # behaves exactly as before (DEFAULT_BUILD_ASSETS / DEFAULT_HOLDOUT_ASSETS).
    build_assets = tuple(args.build_assets) if args.build_assets else C.DEFAULT_BUILD_ASSETS
    holdout_assets = tuple(args.holdout_assets) if args.holdout_assets is not None \
        else C.DEFAULT_HOLDOUT_ASSETS
    cfg = C.CandleConfig(
        window=int(args.window),
        fsq_levels=levels,
        batch=int(args.batch),
        build_assets=build_assets,
        holdout_assets=holdout_assets,
        db=db,
    )
    codebook_size = cfg.codebook_size
    print(f"[cfg] db={db} W={cfg.window} levels={levels} batch={cfg.batch} "
          f"codebook={codebook_size}", flush=True)

    # multi-device divisibility invariant (single-GPU: trivially satisfied).
    if n_dev > 1:
        shard.assert_divisible(cfg.batch, n_dev)

    # ---- model + optimizer ----
    model = L4Tokenizer(window=cfg.window, n_feat=C.N_FEAT, hidden=int(args.hidden),
                        fsq_levels=levels, conv_ch=conv_ch)
    key = jax.random.PRNGKey(int(args.seed))
    dummy = jnp.zeros((cfg.batch, cfg.window, C.N_FEAT), dtype=jnp.float32)
    params = model.init(key, dummy)
    tx = optax.adam(float(args.lr))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # ---- (3)(4) jit train step: donate xb ONLY; block_until_ready after each step ----
    def _train_step(state, xb, valid):
        def loss_fn(p):
            out = state.apply_fn(p, xb)
            return recon_loss(out, xb, valid)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss
    # donate the batch buffer (arg index 1 = xb) ONLY — NEVER state (params/opt_state, arg 0).
    train_step = jax.jit(_train_step, donate_argnums=(1,))

    # ---- (7) frozen encode: idx (+ optional recon for A2). NOT donated (read-only). ----
    @jax.jit
    def _encode_idx(params, xb):
        out = model.apply(params, xb)
        return out["idx"]

    @jax.jit
    def _encode_idx_recon(params, xb):
        out = model.apply(params, xb)
        return out["idx"], out["recon"]

    def apply_encode(params, xb, want_recon=False):
        if want_recon:
            return _encode_idx_recon(params, xb)
        return _encode_idx(params, xb)

    # ---- training loop over the BUILD stream ----
    t0 = time.time()
    global_step = 0
    for ep in range(int(args.epochs)):
        streamer = BatchStreamer(cfg, cfg.build_assets, shuffle=True,
                                 max_windows_per_asset=max_windows)
        ep_loss = 0.0
        nb = 0
        # TASK_796 collapse visibility: accumulate a per-epoch O(codebook) id histogram so we can log
        # effective-code-count (2^entropy of the id distribution) + top1_share EACH epoch. This makes
        # a codebook collapse visible DURING training (the v1 172/12800 disease was only seen post-hoc
        # at eval). One extra (read-only) encode forward per batch; the bincount stays O(codebook).
        ep_id_counts = np.zeros(int(codebook_size), dtype=np.int64)
        try:
            for batch in streamer:
                xb = jax.device_put(np.ascontiguousarray(batch["X"], dtype=np.float32))
                valid = jax.device_put(np.ascontiguousarray(batch["valid"], dtype=bool))
                # encode with CURRENT params BEFORE the donating train step consumes xb (donate_argnums
                # frees xb in-place); _encode_idx does NOT donate, so this read is safe and cheap.
                ids_dev = _encode_idx(state.params, xb)
                state, loss = train_step(state, xb, valid)
                # (4) host must not run ahead of device -> no un-freed async buffer pileup.
                jax.block_until_ready(loss)
                ids = np.asarray(ids_dev)[np.asarray(batch["valid"])]
                if ids.size:
                    ids = np.clip(ids.astype(np.int64), 0, int(codebook_size) - 1)
                    ep_id_counts += np.bincount(ids, minlength=int(codebook_size)).astype(np.int64)
                ep_loss += float(loss)
                nb += 1
                global_step += 1
        finally:
            streamer.close()
        mean_loss = ep_loss / max(nb, 1)
        # effective-code-count = 2^entropy(id_hist); top1_share = mass of the single most-used id.
        ep_total = int(ep_id_counts.sum())
        ep_used = int((ep_id_counts > 0).sum())
        ep_eff = float(2.0 ** EB._entropy_bits_from_counts(ep_id_counts)) if ep_total else 0.0
        ep_top1 = float(ep_id_counts.max() / ep_total) if ep_total else 1.0
        print(f"[train] epoch {ep+1}/{args.epochs}  batches={nb}  recon_mse={mean_loss:.6f}  "
              f"used={ep_used}/{codebook_size}  eff_codes(2^H)={ep_eff:.1f}  top1={ep_top1:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    train_wall = time.time() - t0
    print(f"[train] done: {global_step} steps in {train_wall:.1f}s", flush=True)

    # ---- FREEZE + held-out encode (online bincount, O(codebook)) ----
    frozen_params = state.params
    have_holdout = bool(cfg.holdout_assets) and len(cfg.holdout_assets) > 0
    eval_assets = list(cfg.holdout_assets) if have_holdout else list(cfg.build_assets)
    print(f"[eval] encoding held-out stream over assets={eval_assets} "
          f"(online bincount, codebook={codebook_size})", flush=True)

    counts, n_tokens, a2_pairs, len_counts = _encode_stream_online(
        apply_encode, frozen_params, cfg, eval_assets, codebook_size,
        max_windows_per_asset=max_windows, chunk=int(args.encode_chunk),
    )
    used = int((counts > 0).sum())
    top1 = float(counts.max() / max(counts.sum(), 1)) if counts.sum() > 0 else 1.0
    print(f"[eval] encoded {n_tokens} tokens; codebook used {used}/{codebook_size}; "
          f"top1_share={top1:.4f}", flush=True)

    # ---- eval battery A0 / A1 / A2 / A9 (+ A8 if holdout) ----
    # A0: end-to-end rescale invariance on a tiny diagnostic sample.
    a0_windows = _sample_ohlc_windows(cfg, eval_assets, n=16)
    tokenize_fn = _make_tokenize_window_fn(apply_encode, frozen_params, cfg)
    a0 = EB.a0_rescale_invariance(tokenize_fn, a0_windows, lambdas=(0.01, 100.0)) \
        if a0_windows else dict(agree=float("nan"), n_trials=0, **{"pass": False})
    print(f"[A0] {a0}", flush=True)

    # A1: MDL computed directly from the online O(codebook) histogram. eval_battery.a1_mdl folds an
    # id ITERATOR into exactly this `counts`; since we already folded the stream online, _a1_from_counts
    # computes the identical terms from the histogram and never replays the (O(windows)) stream.
    a1 = _a1_from_counts(counts, codebook_size, n_tokens, cfg.window)
    print(f"[A1] {a1}", flush=True)

    # A2: reconstruction fidelity over the streamed (bounded) pair sample. L4 reconstructs a
    # strictly-POSITIVE price path (start * exp(cumsum(recon_lr))), so mode="ratio" is correct
    # (TASK_796: additive mode is for SIGNED vol-unit move members L1/L2/L3, not L4's price path).
    a2 = EB.a2_reconstruction(iter(a2_pairs), ratio_tol=0.02, mode="ratio") if a2_pairs \
        else dict(frac_within_tol=float("nan"), mean_log_ratio_mse=float("nan"), n_bars=0)
    print(f"[A2] {a2}", flush=True)

    # A9: length non-degeneracy (fixed-length member -> single spike at W, non-degenerate by design,
    # but we ALSO surface the token-id mega-share via top1 so a one-token collapse is visible).
    a9 = EB.a9_length_nondegeneracy(len_counts)
    # augment with the real mega-token share (a9 only sees lengths; top1 over ids is the true test).
    a9 = dict(a9)
    a9["max_token_id_share"] = top1
    a9["id_collapse"] = bool(top1 >= 0.999)
    # TASK_796: codebook-USAGE FLOOR so a future codebook collapse (the v1 172/12800 = 1.3% disease)
    # FAILS A9, not just the degenerate 1-token case. The fixed-length-W stream is non-degenerate by
    # length, and top1<0.999 only catches a single mega-token; a stream that uses (say) 8/256 codes
    # passes both yet is collapsed. used_frac = used_codes/codebook_size with a conservative floor.
    used_frac = float(used) / float(max(int(codebook_size), 1))
    a9["used_codes"] = int(used)
    a9["codebook_size"] = int(codebook_size)
    a9["used_frac"] = used_frac
    a9["used_frac_floor"] = float(USED_FRAC_FLOOR)
    a9["codebook_collapse"] = bool(used_frac < USED_FRAC_FLOOR)
    a9["pass"] = bool(a9.get("pass", False) and not a9["id_collapse"]
                      and not a9["codebook_collapse"])
    a9["pass_"] = a9["pass"]
    print(f"[A9] {a9}", flush=True)

    # A8: cross-asset transfer (only meaningful with a distinct holdout corpus).
    a8 = None
    a8_power = None
    if have_holdout:
        a8 = _a8_transfer(apply_encode, frozen_params, cfg, codebook_size,
                          per_asset_matrix=bool(args.transfer_matrix))
        # TASK_796/797: run BOTH positive-control power harnesses (synthetic, no DB/GPU) and store on
        # the a8 dict so EB.render_a8_lines surfaces the metric-power line and a reader can see the
        # transfer metric itself was valid (unigram + bigram) before trusting the go/no-go.
        a8_power = EB.run_a8_power()
        a8["a8_power"] = a8_power
        print(f"[A8] {a8}", flush=True)
        print(f"[A8.power] {a8_power}", flush=True)

    # ---- RESULT.md + json sidecar ----
    verdict = _verdict(a0, a1, a2, a9, a8)
    sections = _build_sections(a0, a1, a2, a9, a8, used, codebook_size, top1,
                               n_tokens, mean_loss, global_step, train_wall, cfg, levels, conv_ch)
    meta = dict(
        db=cfg.db, window=cfg.window, fsq_levels=list(levels), conv_ch=list(conv_ch),
        hidden=int(args.hidden), batch=cfg.batch, epochs=int(args.epochs), lr=float(args.lr),
        seed=int(args.seed), codebook_size=codebook_size, n_tokens=n_tokens,
        used_codes=used, top1_share=top1, final_recon_mse=float(mean_loss),
        n_devices=n_dev, build_assets=list(cfg.build_assets), holdout_assets=list(cfg.holdout_assets),
        train_wall_s=train_wall,
        a0=a0, a1=a1, a2=a2, a9=a9, a8=a8, a8_power=a8_power,
    )
    write_result(args.out, title="L4 FSQ-VQ candle-language member (TASK_789)",
                 sections=sections, verdict=verdict, meta=meta)
    print(f"[done] verdict={verdict}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# A1 from the online histogram (O(codebook), no stream replay)
# --------------------------------------------------------------------------- #
def _a1_from_counts(counts: np.ndarray, codebook_size: int, n_tokens: int, window: int) -> dict:
    """Compute the eval_battery A1 MDL terms directly from the O(codebook) histogram.

    a1_mdl folds an id ITERATOR into exactly this `counts` accumulator; since we already folded the
    stream online, replaying it would be O(windows). We instead feed a1_mdl a SINGLE chunk that
    re-materializes the histogram support repeated by its counts ONLY if small; otherwise we compute
    the identical terms inline from `counts`. Here we always compute inline (host O(codebook)).

    Null/boundary semantics match the contract: null_bits_per_bar = log2(codebook) per covered bar;
    a fixed-length member emits one boundary per token (n_boundaries == n_tokens); codebook_bits is
    the one-time vocab ship cost (codebook_size * bits-per-code ~ log2(codebook)).

    TASK_796: the per-bar null MUST be charged over the true number of CANDLE BARS the token stream
    covers, NOT over token count. L4 is a fixed-length-W member: each emitted token covers exactly
    `window` raw feature bars, so n_bars_covered = n_tokens * window. Charging the null over token
    count (the v1 bug) structurally pins compression_ratio below 1 because it ignores that one token
    replaces `window` null-bars. We thread n_bars_covered into the per-bar null + the bits_per_candle
    denominator so A1 is a like-for-like per-bar comparison (matching eval_battery.a1_mdl).
    """
    total = int(counts.sum())
    if total <= 0:
        return dict(bits_per_candle=float("nan"), covered_bits=0.0, null_bits=0.0,
                    boundary_bits=0.0, codebook_bits=0.0, entropy_bits_per_token=0.0,
                    n_tokens=0, n_bars_covered=0, codebook_size=int(codebook_size),
                    compression_ratio=float("nan"))
    # fixed-length-W member: every token covers exactly `window` candle bars (TASK_796).
    n_bars_covered = int(total) * int(window)
    p = counts.astype(np.float64) / float(total)
    nz = p[p > 0.0]
    h_bits_per_token = float(-np.sum(nz * np.log2(nz)))
    covered_bits = float(h_bits_per_token * total)
    null_bits_per_bar = float(np.log2(max(int(codebook_size), 2)))
    # per-bar null charged over COVERED BARS, not tokens (TASK_796 fix).
    null_bits = float(null_bits_per_bar * n_bars_covered)
    n_boundaries = int(total)
    boundary_bits = float(n_boundaries * np.log2(max(int(codebook_size), 2)))
    codebook_bits = float(int(codebook_size) * np.log2(max(int(codebook_size), 2)))
    # bits_per_candle denominator is the covered BARS (TASK_796), so a fixed-W member amortizes the
    # token content + boundary + codebook cost over all `window` bars each token covers.
    denom = max(n_bars_covered, 1)
    bits_per_candle = (covered_bits + boundary_bits + codebook_bits) / float(denom)
    return dict(
        bits_per_candle=float(bits_per_candle),
        covered_bits=float(covered_bits),
        null_bits=float(null_bits),
        boundary_bits=float(boundary_bits),
        codebook_bits=float(codebook_bits),
        entropy_bits_per_token=float(h_bits_per_token),
        n_tokens=int(total),
        n_bars_covered=int(n_bars_covered),
        codebook_size=int(codebook_size),
        compression_ratio=float(null_bits / max(covered_bits + boundary_bits + codebook_bits, C.DIV_EPS)),
    )


# --------------------------------------------------------------------------- #
# A8 transfer wired to eval_battery (build vocab = trained model; encode = online ids)
# --------------------------------------------------------------------------- #
def _a8_transfer(apply_encode, params, cfg, codebook_size, per_asset_matrix=False) -> dict:
    """Wire the trained, frozen L4 tokenizer into eval_battery.a8_cross_asset_transfer.

    The 'vocab' is the frozen model itself (its codebook is the alphabet). build_vocab_fn just
    returns a tiny carrier exposing .codebook_size; encode_assets_fn streams token-id chunks for the
    given assets, optionally through the structure-destroying 'shuffle' control (phase-shuffle of the
    vol-unit feature returns, donor data.py:165-203 idiom — applied to the substrate BEFORE encode).
    All folds inside eval_battery are O(codebook); we only ever yield per-batch id chunks here.
    """
    import jax

    # A class body does NOT close over enclosing-function locals (it has its own scope), so
    # `codebook_size = int(codebook_size)` inside `class _Vocab` raises NameError. Capture the
    # value under a DIFFERENT name in the enclosing scope first, then bind it as a class attr.
    cb = int(codebook_size)

    class _Vocab:
        codebook_size = cb

    def build_vocab_fn(assets=None):
        return _Vocab()

    def _phase_shuffle_window(x_bw6, rng):
        """Destroy TEMPORAL structure within each window by phase-shuffling the FFT of each channel
        (donor phase_shuffle_returns idiom). Preserves the marginal/spectral magnitude, kills the
        ordered shape -> a fair same-pipeline null. x_bw6: (B,W,6) float32. This is the "shuffle"
        control (substrate-level structure destruction BEFORE encode)."""
        B, W, F = x_bw6.shape
        xf = np.fft.rfft(x_bw6, axis=1)
        mag = np.abs(xf)
        # random phases shared across the batch axis per (freq,channel) but randomized per call.
        ph = rng.uniform(0.0, 2.0 * np.pi, size=(B, xf.shape[1], F))
        ph[:, 0, :] = 0.0  # keep DC real
        xf_shuf = mag * np.exp(1j * ph)
        return np.fft.irfft(xf_shuf, n=W, axis=1).astype(np.float32)

    # "random_merge" must be a DISTINCT control from "shuffle" (TASK_796): instead of destroying
    # temporal structure, it destroys the CODEBOOK ALIGNMENT — encode the real window, then remap
    # each token id through a FIXED random permutation of [0, codebook). This keeps each member's
    # per-asset id MARGINAL identical to the real encode but breaks the build<->holdout shared-id
    # structure (a given shape no longer maps to the same id across assets in a structured way),
    # so A8's codelength on this control measures "matched marginals, no real id transfer".
    # The permutation is DETERMINISTIC (fixed seed) so build and holdout are remapped consistently.
    _merge_perm = np.random.default_rng(99991).permutation(int(codebook_size)).astype(np.int64)

    def encode_assets_fn(vocab, assets, control):
        rng = np.random.default_rng(12345)
        mwpa = None
        streamer = BatchStreamer(cfg, list(assets), shuffle=False, max_windows_per_asset=mwpa)
        try:
            for batch in streamer:
                xb = np.ascontiguousarray(batch["X"], dtype=np.float32)
                v = np.asarray(batch["valid"])
                if control == "shuffle":
                    xb = _phase_shuffle_window(xb, rng)
                idx = np.asarray(apply_encode(params, jax.numpy.asarray(xb)))
                if control == "random_merge":
                    # remap real ids through the fixed random codebook permutation (id-alignment kill).
                    idx = _merge_perm[np.clip(idx, 0, int(codebook_size) - 1)]
                ids = idx[v]
                if ids.size:
                    yield ids
        finally:
            streamer.close()

    return EB.a8_cross_asset_transfer(
        build_vocab_fn, encode_assets_fn,
        build_assets=cfg.build_assets, holdout_assets=cfg.holdout_assets,
        controls=("shuffle", "random_merge"), exclude_top_k=8,
        per_asset_matrix=per_asset_matrix,
    )


# --------------------------------------------------------------------------- #
# verdict + report sections
# --------------------------------------------------------------------------- #
def _verdict(a0, a1, a2, a9, a8) -> str:
    """A0 (rescale invariance) and A9 (non-degeneracy, incl. the used_codes/codebook_collapse floor
    gates) are HARD gates; A8 (if present) must pass the §186-aligned go/no-go. A1/A2 are reported
    but not pass/fail gates on their own.

    TASK_796/797: the A8 term now gates on EB.a8_verdict(a8) — the FAIR freq-matched random-codebook
    null on BOTH unigram AND bigram axes — NOT the OLD beats_control (== beats_all == beats_random_-
    codebook AND beats_substrate-UNIGRAM), which folded in the entropy-confounded substrate control
    (lore §186, wrong to gate on)."""
    a0_ok = bool(a0.get("pass", False)) if a0 else False
    a9_ok = bool(a9.get("pass", False)) if a9 else False
    a8_ok = EB.a8_verdict(a8)  # None a8 (no holdout) -> True (not applicable)
    if a0_ok and a9_ok and a8_ok:
        return "PASS"
    return "FAIL"


def _a8_lines(a8) -> list:
    """Render the A8 dict via the SHARED eval_battery renderers (TASK_796/797 tech-debt closure).

    The panel COMPUTES the bigram-transfer fields, the per-asset transfer matrix, and the metric
    power harness, but members historically hand-rolled a renderer that only surfaced the UNIGRAM
    headline + the entropy-confounded substrate controls (lore §186). We now delegate to the canonical
    EB.render_a8_lines (unigram + BIGRAM + per-control + power) and EB.render_transfer_matrix
    (per-holdout transfer rate + transferring assets + matrix summary) so every member renders the
    same §186-aligned view. The substrate controls remain REPORTED (not gated)."""
    if not a8:
        return ["(none)"]
    return list(EB.render_a8_lines(a8)) + list(EB.render_transfer_matrix(a8))


def _build_sections(a0, a1, a2, a9, a8, used, codebook_size, top1, n_tokens, mean_loss,
                    steps, train_wall, cfg, levels, conv_ch):
    def _kv(d):
        if not d:
            return ["(none)"]
        return [f"- {k}: {v}" for k, v in d.items()]

    sections = [
        ("Training", [
            f"- steps: {steps}",
            f"- final recon MSE: {mean_loss:.6f}",
            f"- wall: {train_wall:.1f}s",
            f"- window: {cfg.window}  fsq_levels: {list(levels)}  conv_ch: {list(conv_ch)}",
            f"- build_assets: {list(cfg.build_assets)}",
            f"- holdout_assets: {list(cfg.holdout_assets)}",
        ]),
        ("Codebook health", [
            f"- codebook_size: {codebook_size}",
            f"- used_codes: {used} ({100.0*used/max(codebook_size,1):.2f}%)",
            f"- top1_token_share: {top1:.4f}",
            f"- n_tokens_encoded: {n_tokens}",
        ]),
        ("A0 rescale-invariance", _kv(a0)),
        ("A1 MDL", _kv(a1)),
        ("A2 reconstruction", _kv(a2)),
        ("A9 length non-degeneracy", _kv(a9)),
    ]
    if a8 is not None:
        sections.append(("A8 cross-asset transfer", _a8_lines(a8)))
    else:
        sections.append(("A8 cross-asset transfer", ["- skipped (no holdout assets)"]))
    return sections


if __name__ == "__main__":
    raise SystemExit(main())
