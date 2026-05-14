"""Tests for StreamingAssetEncoder (V3) — minGRU temporal encoder."""
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/lukas/Trader')

from mcts_trader.jax_agent.trm_encoder_v3 import (
    StreamingAssetEncoder, FiLMProjector, _mingru_scan,
)


def test_mingru_scan_correctness():
    """Verify parallel scan matches sequential recurrence."""
    key = jax.random.PRNGKey(42)
    N, T, D = 8, 64, 32

    k1, k2 = jax.random.split(key)
    z = jax.nn.sigmoid(jax.random.normal(k1, (N, T, D)))
    h_tilde = jnp.tanh(jax.random.normal(k2, (N, T, D)))

    # Sequential reference
    h = jnp.zeros((N, D))
    for t in range(T):
        h = (1 - z[:, t]) * h + z[:, t] * h_tilde[:, t]
    h_seq = h

    # Parallel (associative scan)
    h_par = _mingru_scan(z, h_tilde)

    max_diff = float(jnp.max(jnp.abs(h_seq - h_par)))
    print(f"Sequential vs parallel max diff: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Mismatch: {max_diff}"
    print("PASS: minGRU scan correctness")


def test_encoder_output_shape():
    """Verify output shape matches V2 interface."""
    key = jax.random.PRNGKey(0)
    B, C, W, A = 4, 27, 64, 64
    state_dim = 64

    encoder = StreamingAssetEncoder(state_dim=state_dim, n_assets=A, embed_dim=32)
    x = jax.random.normal(key, (B, C, W, A))

    params = encoder.init(key, x)
    state = encoder.apply(params, x)

    assert state.shape == (B, state_dim), f"Expected ({B}, {state_dim}), got {state.shape}"
    print(f"PASS: Output shape {state.shape}")

    # Count params
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  V3 parameters: {n_params:,}")


def test_streaming_matches_training():
    """step() output must match __call__() for same input."""
    key = jax.random.PRNGKey(1)
    B, C, W, A = 2, 27, 64, 64
    state_dim = 64

    encoder = StreamingAssetEncoder(state_dim=state_dim, n_assets=A, embed_dim=32)
    x = jax.random.normal(key, (B, C, W, A))

    params = encoder.init(key, x)

    # Training mode (parallel)
    state_train = encoder.apply(params, x)

    # Streaming mode (step by step)
    h = None
    for t in range(W):
        x_t = x[:, :, t, :]  # (B, C, A)
        state_stream, h = encoder.apply(params, x_t, h, method=encoder.step)

    max_diff = float(jnp.max(jnp.abs(state_train - state_stream)))
    print(f"Training vs streaming max diff: {max_diff:.2e}")
    assert max_diff < 1e-3, f"Train/stream mismatch: {max_diff}"
    print("PASS: Streaming matches training")


def test_gradient_flow():
    """Verify non-zero gradients through all parameters."""
    key = jax.random.PRNGKey(2)
    B, C, W, A = 2, 27, 64, 64

    encoder = StreamingAssetEncoder(state_dim=64, n_assets=A, embed_dim=32)
    x = jax.random.normal(key, (B, C, W, A))
    params = encoder.init(key, x)

    def loss_fn(params):
        return jnp.sum(encoder.apply(params, x))

    grads = jax.grad(loss_fn)(params)

    zero_grads = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        norm = float(jnp.linalg.norm(leaf))
        name = '/'.join(str(p) for p in path)
        if norm == 0:
            zero_grads.append(name)

    if zero_grads:
        print(f"FAIL: Zero gradients at: {zero_grads}")
        assert False
    print("PASS: All gradient paths active")


def test_temporal_ordering_matters():
    """Verify that temporal ordering affects output (unlike V2 which flattens)."""
    key = jax.random.PRNGKey(3)
    B, C, W, A = 2, 27, 64, 64

    encoder = StreamingAssetEncoder(state_dim=64, n_assets=A, embed_dim=32)
    x = jax.random.normal(key, (B, C, W, A))
    params = encoder.init(key, x)

    # Forward pass on original
    state_orig = encoder.apply(params, x)

    # Reverse temporal order
    x_rev = x[:, :, ::-1, :]
    state_rev = encoder.apply(params, x_rev)

    diff = float(jnp.mean(jnp.abs(state_orig - state_rev)))
    print(f"Forward vs reversed temporal order: mean diff = {diff:.4f}")
    assert diff > 0.01, f"Temporal reversal should change output significantly, got diff={diff}"
    print("PASS: Temporal ordering matters")


def test_v2_v3_shape_compatibility():
    """V3 is a drop-in replacement for V2 (same input/output shapes)."""
    from mcts_trader.jax_agent.trm_encoder_v2 import AssetTransformerEncoder

    key = jax.random.PRNGKey(4)
    B, C, W, A = 2, 27, 64, 64
    state_dim = 64

    x = jax.random.normal(key, (B, C, W, A))

    # V2
    v2 = AssetTransformerEncoder(state_dim=state_dim, n_assets=A)
    params_v2 = v2.init(key, x)
    out_v2 = v2.apply(params_v2, x)

    # V3
    v3 = StreamingAssetEncoder(state_dim=state_dim, n_assets=A)
    params_v3 = v3.init(key, x)
    out_v3 = v3.apply(params_v3, x)

    assert out_v2.shape == out_v3.shape, f"Shape mismatch: V2={out_v2.shape}, V3={out_v3.shape}"

    n_v2 = sum(p.size for p in jax.tree_util.tree_leaves(params_v2))
    n_v3 = sum(p.size for p in jax.tree_util.tree_leaves(params_v3))
    print(f"V2 params: {n_v2:,}, V3 params: {n_v3:,} (delta: {n_v3 - n_v2:+,})")
    print("PASS: V2/V3 shape compatible")


def test_film_zero_init_byte_identical():
    """TASK_741 (Mint): with zero-init FiLM, output matches baseline exactly."""
    key = jax.random.PRNGKey(5)
    B, C, W, A = 2, 32, 64, 64
    state_dim = 64
    n_s = 5

    x = jax.random.normal(key, (B, C, W, A))
    scalars = jax.random.normal(key, (B, A, n_s))

    base = StreamingAssetEncoder(state_dim=state_dim, n_assets=A, embed_dim=32,
                                 per_asset_scalar_dims=0)
    pb = base.init(key, x)
    out_base = base.apply(pb, x)

    inj = StreamingAssetEncoder(state_dim=state_dim, n_assets=A, embed_dim=32,
                                per_asset_scalar_dims=n_s)
    pi = inj.init(key, x, per_asset_scalars=scalars)
    out_inj = inj.apply(pi, x, per_asset_scalars=scalars)

    max_diff = float(jnp.max(jnp.abs(out_base - out_inj)))
    print(f"FiLM zero-init max diff vs baseline: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Zero-init FiLM must be identity, got {max_diff}"
    print("PASS: FiLM zero-init byte-identical to baseline")


def test_film_nonzero_changes_output():
    """TASK_741: γ/β perturbation shifts encoder output."""
    key = jax.random.PRNGKey(6)
    B, C, W, A = 2, 32, 64, 64
    n_s = 5

    x = jax.random.normal(key, (B, C, W, A))
    scalars = jax.random.normal(key, (B, A, n_s))

    enc = StreamingAssetEncoder(state_dim=64, n_assets=A, embed_dim=32,
                                per_asset_scalar_dims=n_s)
    params = enc.init(key, x, per_asset_scalars=scalars)

    out_zero = enc.apply(params, x, per_asset_scalars=scalars)

    # Perturb film_out kernel
    flat = dict(params['params'])
    film = dict(flat['film_projector'])
    fout = dict(film['film_out'])
    fout['kernel'] = jnp.ones_like(fout['kernel']) * 0.1
    film['film_out'] = fout
    flat['film_projector'] = film
    perturbed = {'params': flat}

    out_pert = enc.apply(perturbed, x, per_asset_scalars=scalars)
    diff = float(jnp.mean(jnp.abs(out_zero - out_pert)))
    print(f"FiLM perturbation mean diff: {diff:.4f}")
    assert diff > 1e-3, f"Perturbing FiLM should change output, got {diff}"
    print("PASS: FiLM perturbation propagates")


def test_film_shape_correctness():
    """TASK_741: FiLMProjector produces correct (γ, β) shapes."""
    key = jax.random.PRNGKey(7)
    B, A, n_s, D = 2, 64, 6, 32
    scalars = jax.random.normal(key, (B, A, n_s))
    film = FiLMProjector(embed_dim=D, hidden_dim=64)
    p = film.init(key, scalars)
    gamma, beta = film.apply(p, scalars)
    assert gamma.shape == (B, A, D), f"γ shape {gamma.shape}"
    assert beta.shape == (B, A, D), f"β shape {beta.shape}"
    print(f"PASS: FiLM γ/β shapes ({B}, {A}, {D})")


if __name__ == '__main__':
    test_mingru_scan_correctness()
    test_encoder_output_shape()
    test_streaming_matches_training()
    test_gradient_flow()
    test_temporal_ordering_matters()
    test_v2_v3_shape_compatibility()
    test_film_zero_init_byte_identical()
    test_film_nonzero_changes_output()
    test_film_shape_correctness()
    print("\n=== ALL TESTS PASSED ===")
