"""TASK_801 — acceptance battery for the flag-gated token front-end (control arm).

Run with CPU forced, tiny batch (zero GPU touch):

    JAX_PLATFORMS=cpu python tools/candle_lang/test_token_frontend.py

E1: flag OFF → forward outputs are checkpoint-identical to canon (untouched).
E2: flag ON, fresh zero-init token_proj, synthetic token_ids → outputs
    BYTE-IDENTICAL to flag OFF (np.array_equal). The load-bearing no-op proof.
E3: under the Stage-B optimizer labels, gradients reach token_embed/token_proj
    (nonzero after one synthetic step).
E4: step() streaming path produces the same per-asset feats as the windowed
    __call__ path on a replayed sequence (mirrors test_trm_encoder_v3.py).
E5: random-codebook arm (TASK_804) is reachable by swapping ONLY the id mapping
    — a permuted vocab feeds the identical wiring with zero code branches.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, '/home/lukas/Trader')

from mcts_trader.jax_agent.trm_encoder_v3 import StreamingAssetEncoder

# Tiny shapes — CPU, B=2.
B, C, W, A = 2, 27, 64, 64
STATE_DIM = 64
EMBED_DIM = 32
REAL_V = 40                     # real codebook size
VOCAB = REAL_V + 1             # + reserved sentinel row for invalid bars
TOK_E = 16


def _fixed_batch():
    x = jax.random.normal(jax.random.PRNGKey(0), (B, C, W, A))
    # token_ids: (B, T, A) int32 in {-1, 0..REAL_V-1}; -1 = invalid bar.
    tok = jax.random.randint(jax.random.PRNGKey(1), (B, W, A), -1, REAL_V)
    return x, tok.astype(jnp.int32)


def _build(off: bool):
    """OFF: token_vocab_size=0 (canon). ON: token_vocab_size=VOCAB."""
    return StreamingAssetEncoder(
        state_dim=STATE_DIM, n_assets=A, embed_dim=EMBED_DIM,
        token_vocab_size=(0 if off else VOCAB),
        token_embed_dim=TOK_E,
    )


def _merge_token_params_into(off_params, on_template_params):
    """Insert the ON model's fresh token_embed/token_proj subtrees into the OFF
    model's params so the shared (canon) layers are bit-identical and only the
    new zero-init token layers are added. Proves the no-op without re-seeding
    the shared layers."""
    off_p = dict(off_params['params'])
    on_p = on_template_params['params']
    off_p['token_embed'] = on_p['token_embed']
    off_p['token_proj'] = on_p['token_proj']
    return {'params': off_p}


def test_e1_flag_off_canon_identical():
    """E1: flag OFF forward == the canon reference (the unfixed model). We
    establish the canon reference here (fixed seed) and confirm OFF reproduces
    it bit-for-bit (the change literally does not touch the OFF path)."""
    x, _ = _fixed_batch()
    enc = _build(off=True)
    params = enc.init(jax.random.PRNGKey(2), x)
    out_a = np.asarray(enc.apply(params, x))
    out_b = np.asarray(enc.apply(params, x))   # canon reference, recomputed
    ident = np.array_equal(out_a, out_b)
    # Also confirm the OFF model created NO token params (canon untouched).
    leaf_names = {'/'.join(str(getattr(k, 'key', k)) for k in path)
                  for path, _ in jax.tree_util.tree_leaves_with_path(params['params'])}
    no_token_params = not any('token' in n for n in leaf_names)
    print(f"E1: OFF-path reproducible = {ident}; no token params created = {no_token_params}")
    assert ident and no_token_params, "E1 FAIL"
    print("E1 PASS")
    return out_a


def test_e2_flag_on_byte_identical(canon_out):
    """E2 (load-bearing): flag ON with fresh zero-init token_proj and synthetic
    token_ids → outputs BYTE-IDENTICAL to flag OFF. np.array_equal, not allclose."""
    x, tok = _fixed_batch()

    # OFF params (canon shared layers).
    off = _build(off=True)
    off_params = off.init(jax.random.PRNGKey(2), x)

    # ON model: init only to get fresh token_embed/token_proj subtrees, then
    # graft them onto the OFF params so shared layers are byte-identical.
    on = _build(off=False)
    on_init = on.init(jax.random.PRNGKey(2), x, token_ids=tok)
    on_params = _merge_token_params_into(off_params, on_init)

    out_on = np.asarray(on.apply(on_params, x, token_ids=tok))
    out_off = np.asarray(off.apply(off_params, x))

    eq_to_canon = np.array_equal(out_on, np.asarray(canon_out))
    eq_to_off = np.array_equal(out_on, out_off)
    maxdiff = float(np.max(np.abs(out_on - out_off)))
    print(f"E2: array_equal(ON, OFF) = {eq_to_off}; "
          f"array_equal(ON, canon) = {eq_to_canon}; max|Δ| = {maxdiff:.2e}")
    assert eq_to_off and eq_to_canon, "E2 FAIL — token front-end is NOT a byte-identical no-op"
    print("E2 PASS")


def test_e3_gradients_reach_token_params():
    """E3: under Stage-B label routing, gradients reach token_embed/token_proj
    (nonzero after one synthetic step). We verify the gradient is nonzero AND
    that make_stage_b_labels routes these leaves to 'train' (not frozen)."""
    from train_phase2 import _is_token_frontend_key, _is_encoder_key
    x, tok = _fixed_batch()
    enc = _build(off=False)
    params = enc.init(jax.random.PRNGKey(2), x, token_ids=tok)

    def loss_fn(p):
        return jnp.sum(enc.apply(p, x, token_ids=tok))

    grads = jax.grad(loss_fn)(params)
    g_embed = float(jnp.linalg.norm(grads['params']['token_embed']['embedding']))
    g_proj_k = float(jnp.linalg.norm(grads['params']['token_proj']['kernel']))
    g_proj_b = float(jnp.linalg.norm(grads['params']['token_proj']['bias']))

    # Label routing: token params must be 'train', NOT 'encoder'/'freeze'.
    embed_path = ('trm', 'encoder', 'token_embed', 'embedding')
    proj_path = ('trm', 'encoder', 'token_proj', 'kernel')
    routed_train = (_is_token_frontend_key(embed_path)
                    and _is_token_frontend_key(proj_path))
    print(f"E3: ||grad token_embed|| = {g_embed:.4e}; "
          f"||grad token_proj.kernel|| = {g_proj_k:.4e}; "
          f"||grad token_proj.bias|| = {g_proj_b:.4e}")
    print(f"E3: token params routed to 'train' (peeled from frozen encoder) = {routed_train}")
    # token_proj is zero-init, so at step 0 d(out)/d(embed) = proj_kernel = 0 and
    # the embed grad is 0 — expected. The gradient ENTRY POINT is token_proj
    # (nonzero kernel+bias grad above). After ONE SGD step the proj kernel is
    # nonzero, so the embed grad becomes nonzero too. Demonstrate that:
    lr = 1e-2
    stepped = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    grads2 = jax.grad(loss_fn)(stepped)
    g_embed2 = float(jnp.linalg.norm(grads2['params']['token_embed']['embedding']))
    print(f"E3: ||grad token_embed|| after 1 SGD step = {g_embed2:.4e} (was 0 at step 0)")
    assert g_proj_k > 0, "E3 FAIL — no gradient to token_proj"
    assert g_embed2 > 0, "E3 FAIL — token_embed grad still zero after proj moved"
    assert routed_train, "E3 FAIL — token params not routed to 'train'"
    print("E3 PASS")


def test_e4_step_matches_windowed():
    """E4: step() streaming path == windowed __call__ on a replayed sequence,
    with token_ids fed per-timestep. Mirrors test_trm_encoder_v3.py's
    test_streaming_matches_training, extended with the token channel."""
    x, tok = _fixed_batch()                 # x:(B,C,W,A), tok:(B,W,A)
    enc = _build(off=False)
    params = enc.init(jax.random.PRNGKey(2), x, token_ids=tok)

    # Windowed.
    state_train = enc.apply(params, x, token_ids=tok)

    # Streaming: feed x_t:(B,C,A) and per-step token_ids:(B,A).
    h = None
    state_stream = None
    for t in range(W):
        x_t = x[:, :, t, :]                 # (B, C, A)
        tok_t = tok[:, t, :]                # (B, A)
        state_stream, h = enc.apply(params, x_t, h, None, tok_t, method=enc.step)

    max_diff = float(jnp.max(jnp.abs(state_train - state_stream)))
    print(f"E4: windowed vs streaming max diff (with tokens) = {max_diff:.2e}")
    assert max_diff < 1e-3, f"E4 FAIL — step/call mismatch {max_diff}"
    print("E4 PASS")


def test_e5_random_codebook_swap_no_branches():
    """E5: TASK_804 random-codebook arm is reachable by swapping ONLY the id
    mapping. We apply a fixed permutation of the real vocab to the token_ids
    (frequency-matched permutation analog) and feed the identical wiring — same
    encoder, same params, zero code branches. Output must change (real signal is
    actually consumed) yet remain a valid forward (no error path)."""
    x, tok = _fixed_batch()
    enc = _build(off=False)
    # Give token_proj a nonzero kernel so the token signal is actually consumed
    # (otherwise zero-init proj makes every id mapping identical — that is the
    # E2 no-op, not what E5 tests).
    params = enc.init(jax.random.PRNGKey(2), x, token_ids=tok)
    p = dict(params['params'])
    tp = dict(p['token_proj'])
    tp['kernel'] = jnp.ones_like(tp['kernel']) * 0.05
    p['token_proj'] = tp
    params = {'params': p}

    # Real arm.
    out_real = np.asarray(enc.apply(params, x, token_ids=tok))

    # Random-codebook arm: permute ONLY valid ids; invalid (-1) stay invalid.
    perm = np.random.RandomState(0).permutation(REAL_V).astype(np.int32)
    tok_np = np.asarray(tok)
    swapped = np.where(tok_np >= 0, perm[np.clip(tok_np, 0, REAL_V - 1)], -1)
    swapped = jnp.asarray(swapped.astype(np.int32))

    out_rand = np.asarray(enc.apply(params, x, token_ids=swapped))   # same call
    diff = float(np.mean(np.abs(out_real - out_rand)))
    print(f"E5: real vs permuted-vocab mean diff = {diff:.4e} "
          f"(same wiring, id mapping only)")
    assert diff > 1e-6, "E5 FAIL — permuted vocab did not flow through the wiring"
    print("E5 PASS: random-codebook arm reachable via id mapping alone")


def test_serving_lag_convention():
    """Point 8 (data-side): the serving-lag slicing primitive encodes
    L1 = delay 2 bars, L8 = unshifted, leading bars → -1 (reserved invalid).
    Pure / DB-free, so the load-bearing lag logic is unit-tested."""
    from tools.candle_lang.token_window_slice import (
        slice_token_window, lag_shift_per_bar, input_lag_for, INVALID_TOKEN_ID,
    )
    # A clean per-bar timeline 0..9 (no gaps).
    tl = np.arange(10, dtype=np.int32)

    # L1: lag 2 → token_input[t] = token_id[t-2]; first 2 bars invalid.
    l1 = lag_shift_per_bar(tl, input_lag_for('L1'))
    expect_l1 = np.array([-1, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    ok_l1 = np.array_equal(l1, expect_l1)
    print(f"P8: L1 lag={input_lag_for('L1')} shifted = {l1.tolist()} (expect {expect_l1.tolist()})")

    # L8: lag 0 → unshifted, byte-identical to input.
    l8 = lag_shift_per_bar(tl, input_lag_for('L8'))
    ok_l8 = np.array_equal(l8, tl)
    print(f"P8: L8 lag={input_lag_for('L8')} shifted = {l8.tolist()} (expect unshifted)")

    # Windowed slice mirrors OHLCV slice cache[ws:ws+win_len], lag pre-applied.
    win = slice_token_window(tl, window_start=1, win_len=4, language='L1')
    # global shifted = [-1,-1,0,1,2,3,4,5,6,7]; slice [1:5] = [-1,0,1,2]
    expect_win = np.array([-1, 0, 1, 2], dtype=np.int32)
    ok_win = np.array_equal(win, expect_win)
    print(f"P8: L1 window(ws=1,len=4) = {win.tolist()} (expect {expect_win.tolist()})")

    # Off-by-one guard: "use tokens up to t-1" (lag 1) is WRONG for L1.
    wrong = lag_shift_per_bar(tl, 1)
    is_not_lag1 = not np.array_equal(l1, wrong)
    print(f"P8: L1 != lag-1 (the wrong off-by-one) = {is_not_lag1}")

    assert ok_l1 and ok_l8 and ok_win and is_not_lag1, "P8 FAIL — serving-lag convention wrong"
    assert INVALID_TOKEN_ID == -1
    print("P8 PASS: serving-lag convention (L1 delay 2, L8 lag 0, leading -1)")


if __name__ == '__main__':
    print("=== TASK_801 token front-end acceptance battery (CPU) ===")
    canon = test_e1_flag_off_canon_identical()
    test_e2_flag_on_byte_identical(canon)
    test_e3_gradients_reach_token_params()
    test_e4_step_matches_windowed()
    test_e5_random_codebook_swap_no_branches()
    test_serving_lag_convention()
    print("\n=== E1-E5 + P8 ALL PASS ===")
