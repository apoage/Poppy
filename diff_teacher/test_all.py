#!/usr/bin/env python3
"""
Comprehensive test suite for diff_teacher module.

Run: python -m diff_teacher.test_all

Tests:
    1. Data Loader - database reading, windowing, normalization
    2. Encoder - VAE forward pass, encoding, decoding
    3. Flow Matching - velocity prediction, sampling
    4. Optimal Solver - hindsight PnL calculation
    5. Regret Curriculum - difficulty adjustment
    6. Teacher Integration - full pipeline
"""

import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    print(f"  {symbol} {name}: {status}")
    if details:
        print(f"    {details}")


def test_data_loader(db_path: str) -> bool:
    """Test data loader functionality."""
    print_header("1. Data Loader Test")

    from .data_loader import MarketDataLoader

    all_passed = True

    try:
        # Initialize
        t0 = time.time()
        loader = MarketDataLoader(
            db_path=db_path,
            n_assets=64,
            window_size=400,
            start_date="2023-01-01",
            end_date="2024-06-30",
        )
        init_time = time.time() - t0

        print_result("Initialize loader", True, f"{init_time:.1f}s")

        # Check basic properties
        n_windows = len(loader)
        print_result("Windows available", n_windows > 0, f"{n_windows:,} windows")
        all_passed &= n_windows > 0

        # Test random window
        window = loader.random_window()
        shape_ok = window.prices.shape == (400, 64, 5)
        print_result("Random window shape", shape_ok, f"{window.prices.shape}")
        all_passed &= shape_ok

        # Test regime detection
        regime_ok = window.regime_label in ["crash", "rally", "sideways", "volatile"]
        print_result("Regime detection", regime_ok, f"regime={window.regime_label}")
        all_passed &= regime_ok

        # Test batch
        prices, timestamps, regimes = loader.get_batch_array(batch_size=8)
        batch_ok = prices.shape == (8, 400, 64, 5)
        print_result("Batch loading", batch_ok, f"{prices.shape}")
        all_passed &= batch_ok

        # Test normalization
        normalized = loader.normalize(prices)
        norm_ok = normalized.min() >= -10 and normalized.max() <= 10
        print_result("Normalization", norm_ok, f"range=[{normalized.min():.2f}, {normalized.max():.2f}]")
        all_passed &= norm_ok

        loader.close()

    except Exception as e:
        print_result("Data loader", False, str(e))
        all_passed = False

    return all_passed


def test_encoder() -> bool:
    """Test encoder (VAE) functionality."""
    print_header("2. Encoder Test")

    from .encoder import MarketEncoder, MarketVAE, vae_loss

    all_passed = True
    key = jax.random.PRNGKey(42)

    try:
        # Test encoder only
        encoder = MarketEncoder(
            latent_dim=128,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )

        # Smaller input for speed
        x = jax.random.normal(key, (2, 100, 32, 5))

        key, init_key = jax.random.split(key)
        params = encoder.init(init_key, x, training=False)

        t0 = time.time()
        latent = encoder.apply(params, x, training=False)
        enc_time = time.time() - t0

        enc_ok = latent.shape == (2, 128)
        print_result("Encoder forward", enc_ok, f"latent={latent.shape}, time={enc_time*1000:.1f}ms")
        all_passed &= enc_ok

        # Count params
        n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
        print_result("Encoder params", True, f"{n_params/1e6:.2f}M")

        # Test VAE
        vae = MarketVAE(
            latent_dim=128,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            window_size=100,
            n_assets=32,
            n_features=5,
        )

        key, init_key, sample_key = jax.random.split(key, 3)
        vae_params = vae.init(init_key, x, sample_key, training=False)

        t0 = time.time()
        recon, mu, log_var = vae.apply(vae_params, x, sample_key, training=False)
        vae_time = time.time() - t0

        vae_ok = recon.shape == x.shape
        print_result("VAE forward", vae_ok, f"recon={recon.shape}, time={vae_time*1000:.1f}ms")
        all_passed &= vae_ok

        # Test loss
        loss, metrics = vae_loss(recon, x, mu, log_var)
        loss_ok = not jnp.isnan(loss) and loss > 0
        print_result("VAE loss", loss_ok, f"loss={float(loss):.4f}")
        all_passed &= loss_ok

        n_vae_params = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
        print_result("VAE params", True, f"{n_vae_params/1e6:.2f}M")

    except Exception as e:
        print_result("Encoder", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


def test_flow_matching() -> bool:
    """Test flow matching generator."""
    print_header("3. Flow Matching Test")

    from .flow_matching import FlowMatchingGenerator, flow_matching_loss

    all_passed = True
    key = jax.random.PRNGKey(42)

    try:
        model = FlowMatchingGenerator(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            ff_dim=256,
        )

        batch_size = 4
        seq_len = 50
        dim = 128  # Latent dim

        x_1 = jax.random.normal(key, (batch_size, seq_len, dim))
        t = jax.random.uniform(key, (batch_size,))
        regime = jnp.array([0, 1, 2, 3])

        key, init_key, noise_key = jax.random.split(key, 3)
        params = model.init(init_key, x_1, t, noise_key, regime, training=False)

        # Forward pass
        t0 = time.time()
        v_pred, v_target, x_t = model.apply(params, x_1, t, noise_key, regime, training=False)
        fwd_time = time.time() - t0

        fwd_ok = v_pred.shape == v_target.shape == x_1.shape
        print_result("Forward pass", fwd_ok, f"v={v_pred.shape}, time={fwd_time*1000:.1f}ms")
        all_passed &= fwd_ok

        # Loss
        loss, metrics = flow_matching_loss(v_pred, v_target)
        loss_ok = not jnp.isnan(loss)
        print_result("Loss computation", loss_ok, f"loss={float(loss):.4f}")
        all_passed &= loss_ok

        # Single-step sampling
        key, sample_key = jax.random.split(key)
        t0 = time.time()
        samples = model.apply(
            params,
            sample_key,
            (batch_size, seq_len, dim),
            regime,
            method=model.sample_single_step,
        )
        sample_time = time.time() - t0

        sample_ok = samples.shape == (batch_size, seq_len, dim)
        print_result("Single-step sample", sample_ok, f"{samples.shape}, time={sample_time*1000:.1f}ms")
        all_passed &= sample_ok

        # Multi-step sampling
        t0 = time.time()
        samples_4 = model.apply(
            params,
            sample_key,
            (batch_size, seq_len, dim),
            regime,
            4,  # 4 steps
            method=model.sample,
        )
        sample4_time = time.time() - t0

        sample4_ok = samples_4.shape == (batch_size, seq_len, dim)
        print_result("4-step sample", sample4_ok, f"{samples_4.shape}, time={sample4_time*1000:.1f}ms")
        all_passed &= sample4_ok

        # Params
        n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
        mem_mb = n_params * 2 / (1024**2)  # FP16
        print_result("Model size", True, f"{n_params/1e6:.2f}M params, ~{mem_mb:.1f}MB FP16")

    except Exception as e:
        print_result("Flow Matching", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


def test_optimal_solver() -> bool:
    """Test optimal solver."""
    print_header("4. Optimal Solver Test")

    from .optimal_solver import OptimalSolver, optimal_pnl_jax, regret_score

    all_passed = True

    try:
        # Generate test prices (random walk)
        np.random.seed(42)
        n_steps, n_assets = 400, 64
        returns = np.random.randn(n_steps - 1, n_assets) * 0.01
        prices = np.zeros((n_steps, n_assets))
        prices[0] = 100
        prices[1:] = 100 * np.exp(np.cumsum(returns, axis=0))

        # Peak-valley solver
        solver = OptimalSolver(algorithm="peak_valley")
        t0 = time.time()
        result = solver.solve(prices)
        pv_time = time.time() - t0

        pv_ok = result.total_pnl > 0 and result.n_trades > 0
        print_result("Peak-valley", pv_ok, f"pnl={result.total_pnl:.4f}, trades={result.n_trades}, time={pv_time*1000:.2f}ms")
        all_passed &= pv_ok

        # DP solver with costs
        solver_dp = OptimalSolver(fee_bps=10, algorithm="dp")
        t0 = time.time()
        result_dp = solver_dp.solve(prices)
        dp_time = time.time() - t0

        dp_ok = result_dp.total_pnl <= result.total_pnl  # Should be less due to costs
        print_result("DP with costs", dp_ok, f"pnl={result_dp.total_pnl:.4f}, time={dp_time*1000:.2f}ms")
        all_passed &= dp_ok

        # JAX version
        prices_jax = jnp.array(prices)
        _ = optimal_pnl_jax(prices_jax)  # Warmup

        t0 = time.time()
        for _ in range(100):
            _ = optimal_pnl_jax(prices_jax)
        jax_time = (time.time() - t0) / 100

        print_result("JAX solver", True, f"time={jax_time*1000:.3f}ms")

        # Batch JAX
        batch_prices = jnp.array(np.random.randn(32, n_steps, n_assets) * 0.01)
        batch_prices = 100 * jnp.exp(jnp.cumsum(batch_prices, axis=1))

        _ = optimal_pnl_jax(batch_prices)  # Warmup
        t0 = time.time()
        for _ in range(100):
            opt = optimal_pnl_jax(batch_prices)
        batch_time = (time.time() - t0) / 100

        batch_ok = opt.shape == (32,)
        print_result("Batch JAX (32)", batch_ok, f"time={batch_time*1000:.3f}ms, per_sample={batch_time/32*1000:.3f}ms")
        all_passed &= batch_ok

        # Regret score
        optimal = jnp.array([0.1, 0.15, 0.2])
        actual = jnp.array([0.05, 0.10, 0.18])
        regret = regret_score(optimal, actual)

        regret_ok = all(regret >= 0) and all(regret <= 2)
        print_result("Regret score", regret_ok, f"regret={regret}")
        all_passed &= regret_ok

    except Exception as e:
        print_result("Optimal Solver", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


def test_curriculum() -> bool:
    """Test regret curriculum controller."""
    print_header("5. Curriculum Test")

    from .regret_curriculum import RegretCurriculum, AdaptiveNoiseSchedule

    all_passed = True

    try:
        curriculum = RegretCurriculum(
            difficulty_init=0.3,
            difficulty_step=0.1,
            patience=5,
        )

        init_ok = curriculum.difficulty == 0.3
        print_result("Initialize", init_ok, f"difficulty={curriculum.difficulty}")
        all_passed &= init_ok

        # Simulate low regret -> should increase difficulty
        for _ in range(10):
            curriculum.update(0.1)  # Low regret

        increased = curriculum.difficulty > 0.3
        print_result("Increase on low regret", increased, f"difficulty={curriculum.difficulty}")
        all_passed &= increased

        # Reset and test high regret
        curriculum.reset()
        curriculum.state.difficulty = 0.7
        for _ in range(10):
            curriculum.update(0.8)  # High regret

        decreased = curriculum.difficulty < 0.7
        print_result("Decrease on high regret", decreased, f"difficulty={curriculum.difficulty}")
        all_passed &= decreased

        # Test regime sampling
        curriculum.state.difficulty = 0.5
        regimes = [curriculum.sample_regime() for _ in range(100)]
        unique = set(regimes)
        regime_ok = len(unique) >= 2
        print_result("Regime sampling", regime_ok, f"sampled: {unique}")
        all_passed &= regime_ok

        # Test noise schedule
        schedule = AdaptiveNoiseSchedule()
        params_low, active_low = schedule.get_progressive_params(0.1)
        params_high, active_high = schedule.get_progressive_params(0.9)

        noise_ok = len(active_high) > len(active_low)
        print_result("Noise schedule", noise_ok, f"low={active_low}, high={active_high}")
        all_passed &= noise_ok

        # Test save/load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        curriculum.save(path)
        curriculum2 = RegretCurriculum()
        curriculum2.load(path)

        load_ok = curriculum2.difficulty == curriculum.difficulty
        print_result("Save/Load", load_ok, f"loaded difficulty={curriculum2.difficulty}")
        all_passed &= load_ok

        Path(path).unlink()  # Cleanup

    except Exception as e:
        print_result("Curriculum", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


def test_teacher_integration(db_path: str) -> bool:
    """Test full teacher integration."""
    print_header("6. Teacher Integration Test")

    from .config import TeacherConfig
    from .teacher import DiffusionTeacher

    all_passed = True

    try:
        # Small config for testing
        config = TeacherConfig(
            db_path=db_path,
            n_assets=16,
            window_size=100,
            latent_dim=64,
            encoder_layers=2,
            encoder_heads=2,
            encoder_ff_dim=128,
            flow_hidden_dim=64,
            flow_layers=2,
            flow_heads=2,
        )

        teacher = DiffusionTeacher(config)
        print_result("Create teacher", True)

        # Initialize training
        t0 = time.time()
        teacher.init_training()
        init_time = time.time() - t0
        print_result("Init training", True, f"time={init_time:.1f}s")

        # Encoder step
        dummy_prices = jax.random.normal(
            jax.random.PRNGKey(0),
            (4, config.window_size, config.n_assets, 5),
        )

        t0 = time.time()
        enc_metrics = teacher.train_encoder_step(dummy_prices)
        enc_time = time.time() - t0

        enc_ok = "loss" in enc_metrics and not np.isnan(enc_metrics["loss"])
        print_result("Encoder step", enc_ok, f"loss={enc_metrics['loss']:.4f}, time={enc_time*1000:.1f}ms")
        all_passed &= enc_ok

        # Generator step
        dummy_latents = jax.random.normal(
            jax.random.PRNGKey(1),
            (4, config.window_size, config.latent_dim),
        )
        dummy_regimes = jnp.array([0, 1, 2, 3])

        t0 = time.time()
        gen_metrics = teacher.train_generator_step(dummy_latents, dummy_regimes)
        gen_time = time.time() - t0

        gen_ok = "loss" in gen_metrics and not np.isnan(gen_metrics["loss"])
        print_result("Generator step", gen_ok, f"loss={gen_metrics['loss']:.4f}, time={gen_time*1000:.1f}ms")
        all_passed &= gen_ok

        # Generation
        t0 = time.time()
        latents, _ = teacher.generate(batch_size=2, regime="volatile")
        gen_sample_time = time.time() - t0

        gen_sample_ok = latents.shape == (2, config.window_size, config.latent_dim)
        print_result("Generate samples", gen_sample_ok, f"{latents.shape}, time={gen_sample_time*1000:.1f}ms")
        all_passed &= gen_sample_ok

        # Curriculum update
        optimal = np.array([0.1, 0.15])
        actual = np.array([0.05, 0.08])
        new_diff, action = teacher.update_curriculum(optimal, actual)

        curr_ok = 0 <= new_diff <= 1
        print_result("Curriculum update", curr_ok, f"difficulty={new_diff}, action={action}")
        all_passed &= curr_ok

        # Metrics
        metrics = teacher.get_metrics()
        print_result("Get metrics", True, f"keys={list(metrics.keys())}")

    except Exception as e:
        print_result("Teacher", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


def test_speed_benchmark() -> bool:
    """Benchmark generation speed."""
    print_header("7. Speed Benchmark")

    from .flow_matching import FlowMatchingGenerator

    all_passed = True
    key = jax.random.PRNGKey(42)

    try:
        # Realistic sizes
        model = FlowMatchingGenerator(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            ff_dim=512,
        )

        batch_size = 32
        seq_len = 400
        dim = 256

        # Initialize
        x_1 = jax.random.normal(key, (batch_size, seq_len, dim))
        t = jax.random.uniform(key, (batch_size,))
        regime = jnp.zeros((batch_size,), dtype=jnp.int32)

        key, init_key, noise_key = jax.random.split(key, 3)
        params = model.init(init_key, x_1, t, noise_key, regime, training=False)

        # JIT compile
        @jax.jit
        def sample_fn(key, params, regime):
            return model.apply(
                params, key, (batch_size, seq_len, dim), regime,
                method=model.sample_single_step,
            )

        # Warmup
        key, sample_key = jax.random.split(key)
        _ = sample_fn(sample_key, params, regime)

        # Benchmark
        n_iters = 50
        t0 = time.time()
        for i in range(n_iters):
            key, sample_key = jax.random.split(key)
            _ = sample_fn(sample_key, params, regime)

        # Wait for async
        jax.block_until_ready(_)
        total_time = time.time() - t0

        time_per_batch = total_time / n_iters
        samples_per_sec = batch_size / time_per_batch
        steps_per_sec = batch_size * seq_len / time_per_batch

        speed_ok = steps_per_sec > 100  # At least 100 steps/sec
        print_result(
            "Generation speed",
            speed_ok,
            f"{samples_per_sec:.0f} samples/sec, {steps_per_sec:.0f} steps/sec"
        )
        all_passed &= speed_ok

        print_result("Time per batch", True, f"{time_per_batch*1000:.1f}ms for {batch_size} samples")

        # Memory estimate
        n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
        mem_fp16 = n_params * 2 / (1024**3)
        print_result("Model memory", True, f"{n_params/1e6:.1f}M params, ~{mem_fp16:.2f}GB FP16")

    except Exception as e:
        print_result("Speed benchmark", False, str(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test diff_teacher module")
    parser.add_argument("--db-path", default="trading_data.db", help="Database path")
    parser.add_argument("--skip-db", action="store_true", help="Skip database tests")
    parser.add_argument("--only", type=str, help="Run only specific test (data, encoder, flow, solver, curriculum, teacher, speed)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  DIFF_TEACHER TEST SUITE")
    print("=" * 60)
    print(f"  Database: {args.db_path}")
    print(f"  JAX devices: {jax.devices()}")

    results = {}

    # Define tests
    tests = {
        "data": lambda: test_data_loader(args.db_path),
        "encoder": test_encoder,
        "flow": test_flow_matching,
        "solver": test_optimal_solver,
        "curriculum": test_curriculum,
        "teacher": lambda: test_teacher_integration(args.db_path),
        "speed": test_speed_benchmark,
    }

    # Filter tests
    if args.only:
        if args.only not in tests:
            print(f"Unknown test: {args.only}")
            print(f"Available: {list(tests.keys())}")
            return 1
        tests = {args.only: tests[args.only]}

    if args.skip_db:
        tests.pop("data", None)
        tests.pop("teacher", None)

    # Run tests
    for name, test_fn in tests.items():
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            results[name] = False

    # Summary
    print_header("SUMMARY")

    total = len(results)
    passed = sum(results.values())

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {name}: {status}")

    print()
    print(f"  Total: {passed}/{total} passed")

    if passed == total:
        print("\n  All tests passed!")
        return 0
    else:
        print(f"\n  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
