import jax
import jax.numpy as jnp
import numpy as np
import time

def measure_bandwidth(size_mb=1024, iterations=10):
    try:
        # Check for GPU
        devices = jax.devices('gpu')
        device = devices[0]
        print(f"🚀 Device: {device.device_kind} (Platform: {device.platform})")
    except Exception as e:
        print(f"❌ GPU not found or JAX cannot access it: {e}")
        return

    # Create host array (pinned memory if possible by allocator, but here standard numpy)
    shape = (size_mb * 1024 * 1024 // 4,) # float32 = 4 bytes
    print(f"📦 Allocating {size_mb} MB Host Array...")
    host_array = np.random.rand(*shape).astype(np.float32)
    
    # Warmup
    print("🔥 Warming up...")
    _ = jax.device_put(host_array, device)
    
    # Measure Host -> Device (H2D)
    print(f"⏱️  Measuring RAM -> VRAM (H2D)...")
    t0 = time.time()
    for _ in range(iterations):
        device_array = jax.device_put(host_array, device)
        device_array.block_until_ready()
    t1 = time.time()
    
    avg_time_h2d = (t1 - t0) / iterations
    bw_h2d = size_mb / avg_time_h2d / 1024 # GB/s
    print(f"   RAM -> VRAM: {bw_h2d:.2f} GB/s ({avg_time_h2d*1000:.1f} ms per transfer)")

    # Measure Device -> Host (D2H)
    print(f"⏱️  Measuring VRAM -> RAM (D2H)...")
    # Ensure device array is ready on device
    device_array = jax.device_put(host_array, device)
    device_array.block_until_ready()
    
    t0 = time.time()
    for _ in range(iterations):
        # np.array() forces copy back to host
        _ = np.array(device_array)
    t1 = time.time()
    
    avg_time_d2h = (t1 - t0) / iterations
    bw_d2h = size_mb / avg_time_d2h / 1024 # GB/s
    print(f"   VRAM -> RAM: {bw_d2h:.2f} GB/s ({avg_time_d2h*1000:.1f} ms per transfer)")

if __name__ == "__main__":
    measure_bandwidth()
