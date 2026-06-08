"""token_lm.py — the first TOKENISED-MODEL experiment for the candle-language panel (branch C).

NEW EXPLORATORY RESEARCH CODE. No flag-gating, no canon risk. This is the GPU job that runs
sandboxed on the server; it does NOT touch the frozen tokenizer contract (tokenizer_api.py) nor any
Poppy/Foxglove training path. It trains a SMALL causal (decoder-only) transformer language model over
the FROZEN candle-token streams and asks one honest question:

    "Is there sequential GRAMMAR in the candle language beyond the order-1 bigram?"

The currency is bits/token (cross-entropy in log2 == codelength == compression == transfer), the same
unit as the rest of the panel (eval_battery A1/A8). We measure the trained LM's HELD-OUT bits/token on
cross-asset HOLDOUT streams it never saw, against two same-data nulls:

  * unigram codelength  (order-0): holdout coded under the BUILD Laplace unigram  -> eval_battery._transfer_codelength
  * bigram  codelength  (order-1): holdout transitions under the BUILD Laplace bigram p(curr|prev)
                                     -> eval_battery._fold_with_bigram + _bigram_codelength

PLUS a SHUFFLED-HOLDOUT control: evaluate the SAME trained LM on per-asset token-shuffled holdout
streams. Shuffling destroys order, so an LM that exploits SEQUENTIAL structure (not just the marginal
frequencies) must do markedly WORSE there — its shuffled bits/token should rise toward the unigram.

VERDICT (PASS):
    held-out LM bits/token  <  bigram bits/token            (higher-order grammar exists)
  AND
    shuffled-holdout LM bits/token  ~=  unigram bits/token   (clearly worse than real held-out;
                                                              the LM is using ORDER, not marginals)

Headline number: the gap (bigram_bits - lm_bits) in bits/token.

SPLIT DISCIPLINE (same as A8): the tokenizer vocab is FIT on BUILD assets, the LM is TRAINED only on
BUILD token streams, and ALL metrics (LM, unigram, bigram, shuffled) are measured on the entirely
HELD-OUT HOLDOUT asset streams. Cross-asset, no leakage.

JAX idioms mirror train_l4.py: env set BEFORE jax import (setdefault MEM_FRACTION, NEVER touch
PREALLOCATE), jit train step donating the batch buffer only, block_until_ready before host transfers,
AdamW + cosine decay, result_writer for the RESULT.md + .json sidecar with the headline numbers in meta.

MEMORY DISCIPLINE (server GPU job):
  * VRAM: rely on JAX's DEFAULT preallocation. We never set XLA_PYTHON_CLIENT_PREALLOCATE and never use
    the platform allocator. XLA_PYTHON_CLIENT_MEM_FRACTION is read from the env (default 0.7 via
    --gpu-frac, only setdefault'd so safe_run/launcher wins if it set it first). Single device only —
    we respect CUDA_VISIBLE_DEVICES (no multi-device sharding). BEFORE the train loop we PRINT the
    VRAM envelope (param count, param+optimizer-state bytes, configured mem fraction) so the operator
    can size the launch.
    VRAM SCALING (the knobs to pick a safe server config):
      - params/optimizer ~ O(vocab*d_model + n_layer*d_model^2)   (Adam keeps 2 moments => ~3x param bytes)
      - activations      ~ O(batch * context_len * d_model * n_layer) for the residual stream, PLUS
                           O(batch * n_head * context_len^2) for the attention score matrices.
      So context_len (block_size) is the quadratic term; batch and d_model are the linear scalers.
      Halving batch or d_model ~halves activation VRAM; halving context_len ~quarters the attention term.
  * RAM: BUILD/HOLDOUT token streams are small int arrays — encoded ONCE at startup into RAM dicts and
    packed into a single (N, context_len+1) int array; minibatches are indexed from that RAM array.
    NO DB read and NO disk I/O inside the train loop (anti-pattern the project forbids).

CPU SMOKE (never run the GPU path on the workstation — display-driving 3090 hard-locks under CUDA):
  JAX_PLATFORMS=cpu python -m tools.candle_lang.token_lm --db trading_data.db \
      --build-assets 1087 1091 353 --holdout-assets 365 1118 --max-windows-per-asset 1500 \
      --d-model 32 --n-layer 2 --n-head 2 --block-size 64 --steps 50 --out /tmp/tokenlm_smoke.md
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from . import config as C
from . import eval_battery as EB
from .result_writer import write_result
from .tokenizer_api import fit_and_save, CandleTokenizer


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a small causal transformer LM over frozen candle-token streams and test "
                    "for sequential grammar beyond the order-1 bigram."
    )
    p.add_argument("--db", type=str, default=None,
                   help="SQLite DB path (default: config.db_path() / $CANDLE_DB).")
    p.add_argument("--out", type=str,
                   default="/home/lukas/Trader/runs/token_lm/L_LM_RESULT.md",
                   help="RESULT.md output path (a .json sidecar is written alongside).")
    p.add_argument("--vocab", type=str, default=None,
                   help="path to an EXISTING frozen-vocab JSON to load; if omitted (or missing) a "
                        "fresh L1-BPE vocab is fit on the BUILD assets and saved next to --out.")
    # ---- asset split (same discipline as A8: fit/train on BUILD, eval on HOLDOUT) ----
    p.add_argument("--build-assets", nargs="+", type=int, default=None,
                   help="explicit BUILD asset ids; default None => config.DEFAULT_BUILD_ASSETS.")
    p.add_argument("--holdout-assets", nargs="+", type=int, default=None,
                   help="explicit HOLDOUT asset ids; default None => config.DEFAULT_HOLDOUT_ASSETS.")
    # ---- substrate / tokenizer ----
    p.add_argument("--window", type=int, default=C.DEFAULT_WINDOW, help="vol-unit window length W.")
    p.add_argument("--max-windows-per-asset", type=int, default=20000,
                   help="cap windows kept per asset (most-recent kept); 0 => unbounded.")
    # ---- BPE fit params (only used when fitting a fresh vocab) ----
    p.add_argument("--bpe-max-merges", type=int, default=512, help="BPE merge cap (fit only).")
    p.add_argument("--bpe-p-min", type=int, default=50, help="BPE pair-count floor (fit only).")
    # ---- transformer hyperparameters ----
    p.add_argument("--n-layer", type=int, default=4, help="number of decoder blocks.")
    p.add_argument("--d-model", type=int, default=128, help="model/embedding width.")
    p.add_argument("--n-head", type=int, default=4, help="attention heads.")
    p.add_argument("--context-len", "--block-size", dest="context_len", type=int, default=128,
                   help="causal context length T (a.k.a. block_size). VRAM scales ~O(B*T^2) for "
                        "attention + O(B*T*d_model) for activations.")
    p.add_argument("--d-ff-mult", type=int, default=4, help="MLP hidden = d_ff_mult * d_model.")
    p.add_argument("--dropout", type=float, default=0.0, help="dropout rate (0 = off, deterministic).")
    # ---- optimization ----
    p.add_argument("--steps", type=int, default=4000, help="number of optimizer steps.")
    p.add_argument("--batch", type=int, default=64, help="training batch size (sequences per step).")
    p.add_argument("--lr", type=float, default=3e-4, help="peak AdamW learning rate.")
    p.add_argument("--warmup", type=int, default=200, help="linear warmup steps for the cosine schedule.")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="global-norm gradient clip (0 = off).")
    p.add_argument("--log-every", type=int, default=200, help="train-loss log cadence (steps).")
    p.add_argument("--eval-batch", type=int, default=256,
                   help="batch size for the held-out bits/token eval forward passes.")
    p.add_argument("--gpu-frac", type=float, default=0.7,
                   help="per-device VRAM fraction; only setdefault'd into "
                        "XLA_PYTHON_CLIENT_MEM_FRACTION (safe_run wins if it already set it). "
                        "We NEVER set XLA_PYTHON_CLIENT_PREALLOCATE.")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed (params + batching + shuffle control).")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# data: frozen tokenizer -> per-asset flat token-id streams
# --------------------------------------------------------------------------- #
def _encode_asset_flat(tok: CandleTokenizer, db: str, asset_id: int, cfg, max_windows) -> np.ndarray:
    """Encode ONE asset to a single flat 1-D np.int64 token stream.

    tok.encode_asset yields one 1-D chunk PER SOURCE WINDOW (each window's BPE-merged token ids);
    we concatenate them in chronological order into the asset's contiguous token sequence. The
    streams of different assets are kept SEPARATE (never concatenated across assets) so neither the
    LM windows nor the bigram fold straddle an asset boundary.
    """
    chunks = list(tok.encode_asset(db, int(asset_id), cfg=cfg,
                                   max_windows_per_asset=max_windows, stride=1))
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(chunks).astype(np.int64)


def _encode_streams(tok, db, assets, cfg, max_windows):
    """Return {asset_id: flat_token_stream} for the given assets (empty streams dropped)."""
    out = {}
    for a in assets:
        s = _encode_asset_flat(tok, db, int(a), cfg, max_windows)
        if s.size > 0:
            out[int(a)] = s
            print(f"[data] asset {a}: {s.size} tokens", flush=True)
        else:
            print(f"[data] asset {a}: EMPTY (skipped)", flush=True)
    return out


def _build_packed_examples(streams: dict, context_len: int) -> np.ndarray:
    """Pack per-asset streams into fixed-length (context_len+1) next-token windows.

    Each row is [t_0 .. t_T] of length context_len+1; the LM is trained to predict t_{i+1} from
    t_{0..i} (inputs = row[:-1], targets = row[1:]). Windows NEVER straddle an asset boundary
    (we slide WITHIN each asset's stream only), matching the no-cross-asset-coupling contract.
    Returns an (N, context_len+1) int64 array (may be empty if every stream is too short).
    """
    T1 = int(context_len) + 1
    rows = []
    for s in streams.values():
        if s.size < T1:
            continue
        # non-overlapping packing keeps examples decorrelated and the dataset bounded;
        # a contiguous chop of the asset stream into context_len+1 blocks.
        n = s.size // T1
        rows.append(s[: n * T1].reshape(n, T1))
    if not rows:
        return np.zeros((0, T1), dtype=np.int64)
    return np.concatenate(rows, axis=0).astype(np.int64)


# --------------------------------------------------------------------------- #
# baselines: unigram + bigram codelength on the HELD-OUT streams (eval_battery helpers)
# --------------------------------------------------------------------------- #
def _baseline_codelengths(build_streams: dict, hold_streams: dict, codebook_size: int):
    """Order-0 (unigram) and order-1 (bigram) bits/token of the HOLDOUT streams under a BUILD-fit
    Laplace model, computed with the canonical eval_battery helpers.

    unigram: build a unigram histogram over BUILD, a unigram histogram over HOLDOUT, then
             EB._transfer_codelength with an ALL-TRUE mask == bits/token of holdout under the build
             Laplace unigram (NO megatoken band exclusion here — we want the honest full-vocab number
             to compare the LM against, not the A8 non-saturating headline).
    bigram:  per-asset EB._fold_with_bigram on BUILD (summed transition matrix) and on HOLDOUT, then
             EB._bigram_codelength == bits/transition of holdout transitions under build p(curr|prev).
    """
    K = int(codebook_size)

    # ----- unigram: fold each stream into a (K,) histogram (per-asset, then sum) -----
    build_uni = np.zeros(K, dtype=np.int64)
    hold_uni = np.zeros(K, dtype=np.int64)
    for s in build_streams.values():
        EB._bincount_into(build_uni, s.astype(np.int64), K)
    for s in hold_streams.values():
        EB._bincount_into(hold_uni, s.astype(np.int64), K)
    full_mask = np.ones(K, dtype=bool)
    unigram_bits, n_uni = EB._transfer_codelength(build_uni, hold_uni, full_mask, alpha=1.0)

    # ----- bigram: per-asset transition fold (one ordered pass per asset), summed across assets -----
    # _fold_with_bigram tracks the last id ACROSS chunks WITHIN one iterator, so we feed each asset's
    # stream as its OWN single-chunk iterator -> no cross-asset transition leak.
    build_big = np.zeros((K, K), dtype=np.int64) if K <= EB._BIGRAM_MAX_K else None
    hold_big = np.zeros((K, K), dtype=np.int64) if K <= EB._BIGRAM_MAX_K else None
    if build_big is not None:
        for s in build_streams.values():
            _u, b = EB._fold_with_bigram([s.astype(np.int64)], K)
            if b is not None:
                build_big += b
        for s in hold_streams.values():
            _u, b = EB._fold_with_bigram([s.astype(np.int64)], K)
            if b is not None:
                hold_big += b
        bigram_bits, n_big = EB._bigram_codelength(build_big, hold_big, alpha=1.0)
    else:
        bigram_bits, n_big = float("inf"), 0

    return (float(unigram_bits), int(n_uni)), (float(bigram_bits), int(n_big))


# --------------------------------------------------------------------------- #
# tiny self-contained causal transformer (Flax)
# --------------------------------------------------------------------------- #
def _make_model(vocab_size, n_layer, d_model, n_head, context_len, d_ff, dropout):
    """Construct a small decoder-only transformer (flax.linen). Returns the nn.Module instance.

    Token embedding + learned positional embedding -> n_layer pre-norm blocks (causal multi-head
    self-attention + GELU MLP) -> final LayerNorm -> tied-free output projection to vocab logits.
    Standard next-token softmax (cross-entropy == codelength).
    """
    import jax.numpy as jnp
    import flax.linen as nn

    class Block(nn.Module):
        d_model: int
        n_head: int
        d_ff: int
        dropout: float

        @nn.compact
        def __call__(self, x, mask, deterministic):
            h = nn.LayerNorm()(x)
            h = nn.SelfAttention(
                num_heads=self.n_head, qkv_features=self.d_model, dropout_rate=self.dropout,
                deterministic=deterministic,
            )(h, mask=mask)
            x = x + h
            h = nn.LayerNorm()(x)
            h = nn.Dense(self.d_ff)(h)
            h = nn.gelu(h)
            h = nn.Dense(self.d_model)(h)
            h = nn.Dropout(rate=self.dropout, deterministic=deterministic)(h)
            return x + h

    class CausalLM(nn.Module):
        vocab_size: int
        n_layer: int
        d_model: int
        n_head: int
        context_len: int
        d_ff: int
        dropout: float

        @nn.compact
        def __call__(self, idx, deterministic=True):
            B, T = idx.shape
            tok = nn.Embed(self.vocab_size, self.d_model, name="tok_emb")(idx)         # (B,T,D)
            pos = self.param("pos_emb",
                             nn.initializers.normal(stddev=0.02),
                             (1, self.context_len, self.d_model))
            x = tok + pos[:, :T, :]
            # causal mask (1 = attend): lower-triangular (B,1,T,T).
            causal = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]
            for _ in range(self.n_layer):
                x = Block(self.d_model, self.n_head, self.d_ff, self.dropout)(
                    x, causal, deterministic)
            x = nn.LayerNorm()(x)
            logits = nn.Dense(self.vocab_size, name="head")(x)                          # (B,T,V)
            return logits

    return CausalLM(vocab_size=int(vocab_size), n_layer=int(n_layer), d_model=int(d_model),
                    n_head=int(n_head), context_len=int(context_len), d_ff=int(d_ff),
                    dropout=float(dropout))


def _bits_per_token_from_logits(logits, targets):
    """Mean cross-entropy in LOG2 (bits/token) for next-token prediction. jnp arrays in/out."""
    import jax.nn as jnn
    import jax.numpy as jnp
    logp = jnn.log_softmax(logits, axis=-1)                      # natural log
    tgt = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]   # (B,T) nats
    nats = -jnp.mean(tgt)
    return nats / jnp.log(2.0)                                   # -> bits


# --------------------------------------------------------------------------- #
# held-out bits/token evaluation (batched forward over packed examples)
# --------------------------------------------------------------------------- #
def _eval_bits_per_token(eval_fn, params, examples: np.ndarray, eval_batch: int):
    """Mean bits/token of the trained LM over packed (N, T+1) examples. Token-weighted mean across
    batches. examples already packed (no asset straddle); inputs=row[:-1], targets=row[1:]."""
    import jax
    import jax.numpy as jnp

    if examples.shape[0] == 0:
        return float("nan"), 0
    total_bits = 0.0
    total_tok = 0
    N = examples.shape[0]
    for i in range(0, N, int(eval_batch)):
        block = examples[i:i + int(eval_batch)]
        xb = jnp.asarray(block[:, :-1])
        yb = jnp.asarray(block[:, 1:])
        bits = eval_fn(params, xb, yb)
        jax.block_until_ready(bits)
        ntok = int(yb.shape[0] * yb.shape[1])
        total_bits += float(bits) * ntok
        total_tok += ntok
    return float(total_bits / max(total_tok, 1)), int(total_tok)


def _shuffle_streams(streams: dict, seed: int) -> dict:
    """Per-asset token shuffle (destroys ORDER, preserves the per-asset MARGINAL exactly)."""
    rng = np.random.default_rng(int(seed))
    out = {}
    for a, s in streams.items():
        sh = s.copy()
        rng.shuffle(sh)
        out[int(a)] = sh
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)

    # ---- VRAM env: setdefault MEM_FRACTION only; NEVER touch PREALLOCATE (mirror train_l4.py) ----
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.gpu_frac))

    db = args.db if args.db else C.db_path()
    build_assets = tuple(args.build_assets) if args.build_assets else C.DEFAULT_BUILD_ASSETS
    holdout_assets = tuple(args.holdout_assets) if args.holdout_assets is not None \
        else C.DEFAULT_HOLDOUT_ASSETS
    max_windows = None if int(args.max_windows_per_asset) <= 0 else int(args.max_windows_per_asset)
    cfg = C.CandleConfig(window=int(args.window), build_assets=build_assets,
                         holdout_assets=holdout_assets, db=db)

    print(f"[cfg] db={db} W={cfg.window} build={list(build_assets)} holdout={list(holdout_assets)}",
          flush=True)

    # ---- 1) frozen L1-BPE tokenizer: load existing or fit+save a fresh one on BUILD ----
    vocab_path = args.vocab
    if vocab_path and os.path.exists(vocab_path):
        print(f"[tok] loading frozen vocab {vocab_path}", flush=True)
        tok = CandleTokenizer.from_path(vocab_path)
    else:
        if not vocab_path:
            base = args.out[:-3] if args.out.endswith(".md") else args.out
            vocab_path = base + "_vocab.json"
        print(f"[tok] fitting fresh L1-BPE vocab on BUILD -> {vocab_path}", flush=True)
        _vocab, vocab_path = fit_and_save(
            cfg, build_assets, vocab_path,
            max_windows_per_asset=(max_windows or 20000),
            p_min=int(args.bpe_p_min), max_merges=int(args.bpe_max_merges), verbose=True,
        )
        tok = CandleTokenizer.from_path(vocab_path)
    codebook_size = int(tok.codebook_size)
    print(f"[tok] codebook_size={codebook_size}", flush=True)

    # ---- 2) encode BUILD + HOLDOUT to per-asset flat token streams ----
    print("[data] encoding BUILD streams ...", flush=True)
    build_streams = _encode_streams(tok, db, build_assets, tok.cfg, max_windows)
    print("[data] encoding HOLDOUT streams ...", flush=True)
    hold_streams = _encode_streams(tok, db, holdout_assets, tok.cfg, max_windows)
    if not build_streams:
        raise RuntimeError("token_lm: no BUILD token streams encoded — check db/assets/window.")
    if not hold_streams:
        raise RuntimeError("token_lm: no HOLDOUT token streams encoded — check db/assets/window.")

    # ---- 3) baselines (unigram + bigram) on the HELD-OUT streams ----
    (unigram_bits, n_uni), (bigram_bits, n_big) = _baseline_codelengths(
        build_streams, hold_streams, codebook_size)
    print(f"[baseline] unigram bits/token={unigram_bits:.4f} (n={n_uni})  "
          f"bigram bits/token={bigram_bits:.4f} (n={n_big})", flush=True)

    # ---- 4) packed next-token training examples from BUILD only ----
    train_ex = _build_packed_examples(build_streams, args.context_len)
    if train_ex.shape[0] == 0:
        raise RuntimeError(
            f"token_lm: no BUILD example fits context_len+1={args.context_len + 1}; "
            f"streams too short. Lower --context-len or raise --max-windows-per-asset.")
    print(f"[data] packed {train_ex.shape[0]} training windows of len {args.context_len + 1}",
          flush=True)
    hold_ex = _build_packed_examples(hold_streams, args.context_len)
    print(f"[data] packed {hold_ex.shape[0]} held-out eval windows", flush=True)
    if hold_ex.shape[0] == 0:
        raise RuntimeError("token_lm: no HELD-OUT example fits the context window — lower --context-len.")

    # ---- JAX imports AFTER env set ----
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state

    print("JAX devices:", jax.devices(), flush=True)

    d_ff = int(args.d_ff_mult) * int(args.d_model)
    model = _make_model(codebook_size, args.n_layer, args.d_model, args.n_head,
                        args.context_len, d_ff, args.dropout)

    key = jax.random.PRNGKey(int(args.seed))
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, int(args.context_len)), dtype=jnp.int32)
    params = model.init({"params": init_key, "dropout": init_key}, dummy, deterministic=True)
    n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print(f"[model] n_params={n_params} (n_layer={args.n_layer} d_model={args.d_model} "
          f"n_head={args.n_head} ctx={args.context_len} vocab={codebook_size})", flush=True)

    # ---- VRAM ENVELOPE (printed BEFORE the train loop so the operator can size the launch) ----
    # params are float32 (4 bytes). AdamW keeps two moment buffers (m, v) of param shape, and the
    # TrainState holds the live params => ~3x param bytes resident for params+optimizer state (the
    # transient `updates` tree is donated). Activations are NOT counted here (they scale with
    # batch * context_len^2 * n_head — see the header VRAM SCALING note).
    param_dtype_bytes = 4
    param_bytes = n_params * param_dtype_bytes
    opt_state_bytes = 2 * param_bytes                       # Adam m + v
    param_plus_opt_bytes = param_bytes + opt_state_bytes    # ~3x param bytes
    mem_frac_env = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.gpu_frac))
    print(
        f"[vram] envelope: n_params={n_params:,}  param_bytes={param_bytes/1e6:.2f}MB  "
        f"opt_state(Adam m+v)={opt_state_bytes/1e6:.2f}MB  "
        f"param+opt_state≈{param_plus_opt_bytes/1e6:.2f}MB  "
        f"(activations extra, scale with batch*ctx^2*n_head)", flush=True)
    print(
        f"[vram] XLA_PYTHON_CLIENT_MEM_FRACTION={mem_frac_env}  PREALLOCATE=<default,untouched>  "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}  "
        f"devices={jax.devices()}", flush=True)

    # ---- optimizer: AdamW + cosine decay (warmup -> cosine) + optional global-norm clip ----
    # warmup must leave room for the cosine phase: optax's decay_steps is the TOTAL horizon and it
    # subtracts warmup internally, so warmup >= steps yields a negative decay span. Clamp warmup to
    # < steps (keep >=1 cosine step) — matters for tiny smoke runs (steps<warmup).
    warmup_steps = min(int(args.warmup), max(int(args.steps) - 1, 0))
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=float(args.lr), warmup_steps=warmup_steps,
        decay_steps=int(args.steps), end_value=float(args.lr) * 0.1,
    )
    chain = []
    if float(args.grad_clip) > 0.0:
        chain.append(optax.clip_by_global_norm(float(args.grad_clip)))
    chain.append(optax.adamw(learning_rate=sched, weight_decay=float(args.weight_decay)))
    tx = optax.chain(*chain)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # ---- jit train step (donate the batch buffer xb,yb; never params/opt_state) ----
    def _loss_fn(p, xb, yb, dkey):
        logits = model.apply(p, xb, deterministic=(float(args.dropout) == 0.0),
                             rngs={"dropout": dkey})
        return _bits_per_token_from_logits(logits, yb)

    def _train_step(state, xb, yb, dkey):
        loss, grads = jax.value_and_grad(_loss_fn)(state.params, xb, yb, dkey)
        return state.apply_gradients(grads=grads), loss
    train_step = jax.jit(_train_step, donate_argnums=(1, 2))

    # ---- jit eval step (read-only; bits/token over a batch) ----
    @jax.jit
    def _eval_step(params, xb, yb):
        logits = model.apply(params, xb, deterministic=True)
        return _bits_per_token_from_logits(logits, yb)

    # ---- training loop: random minibatches of packed BUILD windows ----
    N = train_ex.shape[0]
    B = int(args.batch)
    t0 = time.time()
    first_loss = None
    last_loss = None
    rng = np.random.default_rng(int(args.seed) + 1)
    for step in range(int(args.steps)):
        idx = rng.integers(0, N, size=B)
        block = train_ex[idx]
        xb = jnp.asarray(block[:, :-1])
        yb = jnp.asarray(block[:, 1:])
        key, dkey = jax.random.split(key)
        state, loss = train_step(state, xb, yb, dkey)
        jax.block_until_ready(loss)
        lf = float(loss)
        if first_loss is None:
            first_loss = lf
        last_loss = lf
        if (step + 1) % int(args.log_every) == 0 or step == 0:
            print(f"[train] step {step+1}/{args.steps}  train_bits/token={lf:.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    train_wall = time.time() - t0
    loss_decreased = bool(first_loss is not None and last_loss is not None and last_loss < first_loss)
    print(f"[train] done: {args.steps} steps in {train_wall:.1f}s  "
          f"first={first_loss:.4f} last={last_loss:.4f} decreased={loss_decreased}", flush=True)

    # ---- 5) held-out LM bits/token (the metric) ----
    lm_bits, lm_ntok = _eval_bits_per_token(_eval_step, state.params, hold_ex, int(args.eval_batch))
    print(f"[eval] HELD-OUT LM bits/token={lm_bits:.4f} (n={lm_ntok})", flush=True)

    # ---- 6) shuffled-holdout control: same LM, per-asset token-shuffled holdout ----
    shuf_streams = _shuffle_streams(hold_streams, int(args.seed) + 7)
    shuf_ex = _build_packed_examples(shuf_streams, args.context_len)
    shuffled_lm_bits, shuf_ntok = _eval_bits_per_token(
        _eval_step, state.params, shuf_ex, int(args.eval_batch))
    print(f"[eval] SHUFFLED-HOLDOUT LM bits/token={shuffled_lm_bits:.4f} (n={shuf_ntok})", flush=True)

    # ---- verdict ----
    beats_bigram = bool(np.isfinite(lm_bits) and np.isfinite(bigram_bits) and lm_bits < bigram_bits)
    # "shuffled ~= unigram": shuffled LM should be much worse than the real held-out LM and close to
    # the unigram (it can only exploit marginals once order is destroyed). We require BOTH:
    #   (a) shuffled is clearly WORSE than the real held-out LM (the order signal is real), and
    #   (b) shuffled sits near/above the unigram (it collapsed toward the order-0 marginal model).
    shuf_worse_than_real = bool(np.isfinite(shuffled_lm_bits) and np.isfinite(lm_bits)
                                and shuffled_lm_bits > lm_bits + 0.05)
    shuf_near_unigram = bool(np.isfinite(shuffled_lm_bits) and np.isfinite(unigram_bits)
                             and shuffled_lm_bits >= unigram_bits - 0.25)
    order_control_ok = bool(shuf_worse_than_real and shuf_near_unigram)
    verdict = "PASS" if (beats_bigram and order_control_ok) else "FAIL"

    lm_vs_bigram_gap = float(bigram_bits - lm_bits) if (np.isfinite(lm_bits) and np.isfinite(bigram_bits)) else float("nan")
    lm_vs_unigram_gap = float(unigram_bits - lm_bits) if (np.isfinite(lm_bits) and np.isfinite(unigram_bits)) else float("nan")

    print(f"[verdict] {verdict}  lm={lm_bits:.4f}  bigram={bigram_bits:.4f}  unigram={unigram_bits:.4f}  "
          f"shuffled={shuffled_lm_bits:.4f}  gap(bigram-lm)={lm_vs_bigram_gap:+.4f}", flush=True)

    # ---- RESULT.md + json sidecar ----
    sections = [
        ("Setup", [
            f"- db: {db}",
            f"- window: {cfg.window}   codebook_size: {codebook_size}",
            f"- build_assets: {list(build_assets)}",
            f"- holdout_assets: {list(holdout_assets)}",
            f"- vocab: {vocab_path}",
            f"- max_windows_per_asset: {max_windows}",
        ]),
        ("Model", [
            f"- n_layer: {args.n_layer}  d_model: {args.d_model}  n_head: {args.n_head}  "
            f"context_len: {args.context_len}  d_ff: {d_ff}",
            f"- n_params: {n_params}",
            f"- steps: {args.steps}  batch: {args.batch}  lr: {args.lr}  warmup: {args.warmup}  "
            f"weight_decay: {args.weight_decay}  grad_clip: {args.grad_clip}",
            f"- train_windows: {train_ex.shape[0]}  heldout_windows: {hold_ex.shape[0]}",
            f"- train wall: {train_wall:.1f}s   first_bits: {first_loss:.4f}  last_bits: {last_loss:.4f}  "
            f"decreased: {loss_decreased}",
        ]),
        ("Held-out bits/token (lower = better)", [
            f"- **LM (held-out)**: {lm_bits:.4f}   (n_tokens={lm_ntok})",
            f"- bigram (order-1 null): {bigram_bits:.4f}   (n={n_big})",
            f"- unigram (order-0 null): {unigram_bits:.4f}   (n={n_uni})",
            f"- LM on SHUFFLED held-out (order destroyed): {shuffled_lm_bits:.4f}   (n={shuf_ntok})",
        ]),
        ("Grammar test", [
            f"- LM beats bigram (higher-order grammar exists): {beats_bigram}  "
            f"(gap bigram-lm = {lm_vs_bigram_gap:+.4f} bits/token)",
            f"- LM beats unigram: gap unigram-lm = {lm_vs_unigram_gap:+.4f} bits/token",
            f"- shuffled worse than real held-out (uses ORDER): {shuf_worse_than_real}",
            f"- shuffled ~ unigram (collapsed to marginals): {shuf_near_unigram}",
            f"- order control OK: {order_control_ok}",
        ]),
    ]
    meta = dict(
        db=db, window=cfg.window, codebook_size=codebook_size,
        build_assets=list(build_assets), holdout_assets=list(holdout_assets),
        vocab_path=vocab_path, max_windows_per_asset=max_windows,
        n_layer=int(args.n_layer), d_model=int(args.d_model), n_head=int(args.n_head),
        context_len=int(args.context_len), d_ff=int(d_ff), n_params=n_params,
        param_bytes=int(param_bytes), opt_state_bytes=int(opt_state_bytes),
        param_plus_opt_bytes=int(param_plus_opt_bytes), mem_fraction=str(mem_frac_env),
        steps=int(args.steps), batch=int(args.batch), lr=float(args.lr), warmup=int(args.warmup),
        warmup_effective=int(warmup_steps),
        weight_decay=float(args.weight_decay), grad_clip=float(args.grad_clip),
        seed=int(args.seed), train_windows=int(train_ex.shape[0]),
        heldout_windows=int(hold_ex.shape[0]), train_wall_s=float(train_wall),
        first_train_bits=float(first_loss), last_train_bits=float(last_loss),
        loss_decreased=loss_decreased,
        # ---- HEADLINE numbers ----
        lm_bits=float(lm_bits), unigram_bits=float(unigram_bits), bigram_bits=float(bigram_bits),
        shuffled_lm_bits=float(shuffled_lm_bits),
        lm_vs_bigram_gap=lm_vs_bigram_gap, lm_vs_unigram_gap=lm_vs_unigram_gap,
        beats_bigram=beats_bigram, shuffled_worse_than_real=shuf_worse_than_real,
        shuffled_near_unigram=shuf_near_unigram, order_control_ok=order_control_ok,
        lm_ntok=int(lm_ntok), shuffled_ntok=int(shuf_ntok),
        verdict=verdict,
    )
    write_result(args.out,
                 title="Candle-language token LM — sequential grammar beyond order-1 (branch C)",
                 sections=sections, verdict=verdict, meta=meta)
    print(f"[done] verdict={verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
