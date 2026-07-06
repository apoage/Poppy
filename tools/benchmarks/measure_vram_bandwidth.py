import jax
import jax.numpy as jnp
import time

def measure_vram_bandwidth(size_mb=1024, iterations=100):
    try:
        devices = jax.devices('gpu')
        device = devices[0]
        print(f"🚀 Device: {device.device_kind}")
    except:
        print("❌ GPU not found.")
        return

    # 1. Allocate VRAM (float32 = 4 bytes)
    # 1024 MB = 256M elements
    elements = size_mb * 1024 * 1024 // 4
    print(f"📦 Allocating 3x {size_mb} MB arrays in VRAM...")
    
    # Initialize keys and create arrays directly on device
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (elements,)).block_until_ready()
    y = jax.random.normal(key, (elements,)).block_until_ready()
    
    # JIT compile the kernel
    @jax.jit
    def memory_bound_kernel(a, b):
        return a + b

    # Warmup (compilation)
    print("🔥 Warming up...")
    z = memory_bound_kernel(x, y).block_until_ready()

    # Benchmark
    print(f"⏱️  Measuring VRAM Bandwidth (z = x + y)...")
    t0 = time.time()
    for _ in range(iterations):
        z = memory_bound_kernel(x, y)
        z.block_until_ready()
    t1 = time.time()

    # Calculate metrics
    total_time = t1 - t0
    # Data moved: Read x (size) + Read y (size) + Write z (size) = 3 * size
    total_data_gb = (3 * size_mb * iterations) / 1024
    bandwidth = total_data_gb / total_time

    print(f"   Time: {total_time:.4f} s for {iterations} iterations")
    print(f"   Data Moved: {total_data_gb:.2f} GB")
    print(f"   🚀 Effective VRAM Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    measure_vram_bandwidth()
